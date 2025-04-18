import os
import json
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Type, Iterable, Union, Tuple
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_model, load_model

from models.mlpgpt import MLP_GPT
from models.gpt import MLP
from models.sparsified import SparsifiedGPTOutput
from models.mlpsparsified import MLPSparsifiedGPT
from config.sae.models import SAEConfig
from config.sae.training import LossCoefficients
from models.sae import SparseAutoencoder, EncoderOutput 

import einops

def get_jacobian(
    sae_mlpin : SparseAutoencoder,
    sae_mlpout : SparseAutoencoder,
    mlp: MLP,
    topk_indices_mlpin: torch.Tensor,
    topk_indices_mlpout: torch.Tensor,
    mlp_act_grads: torch.Tensor,
) -> torch.Tensor:
    # required to transpose mlp weights as nn.Linear stores them backwards
    # everything should be of shape (d_out, d_in)
    wd1 = sae_mlpin.W_dec @ mlp.W_in.T
    w2e = mlp.W_out.T @ sae_mlpout.W_enc

    jacobian = einops.einsum(
        wd1[topk_indices_mlpin],
        mlp_act_grads,
        w2e[:, topk_indices_mlpout],
        # "... seq_pos k1 d_mlp, ... seq_pos d_mlp,"
        # "d_mlp ... seq_pos k2 -> ... seq_pos k2 k1",
        "... k1 d_mlp, ... d_mlp, d_mlp ... k2 -> ... k2 k1",
    )
    return jacobian


class JSparsifiedGPT(MLPSparsifiedGPT):
    def __init__(
        self, 
        config: SAEConfig,
        loss_coefficients: Optional[LossCoefficients] = None,
        trainable_layers: Optional[tuple] = None,
    ):
        super().__init__(config, loss_coefficients, trainable_layers)
        
    def post_init(self):
        # Replace MLP forward method with our custom implementation
        for layer_idx in self.layer_idxs:
            mlp = self.gpt.transformer.h[layer_idx].mlp
            bound_method = types.MethodType(self.mlp_forward_with_grads, mlp)
            mlp.forward = bound_method
            print(f"Patched MLP forward for layer {layer_idx}")

    def mlp_forward_with_grads(self, 
        module: MLP,
        input: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        print(f"MLP forward with grads")
        pre_actfn = module.c_fc(input)
        post_actfn = module.gelu(pre_actfn)
        output = module.c_proj(post_actfn)
            
        grad_of_actfn = torch.autograd.grad(
            outputs=post_actfn, inputs=pre_actfn,
            grad_outputs=torch.ones_like(post_actfn), retain_graph=True
        )[0]
        print(f"Grad of actfn: {grad_of_actfn}")
        return output, grad_of_actfn
    
    @contextmanager
    def record_activations(self):
        """
        Context manager for recording residual stream activations.

        :yield activations: Dictionary of activations.
        activations[f'{layer_idx}_mlpin'] = h[layer_idx].mlpin
        activations[f'{layer_idx}_mlpout'] = h[layer_idx].mlpout
        # NOTE: resid_mid is stored in self.resid_mid_cache, not yielded directly
        """
        activations: dict[str, torch.Tensor] = {}

        # Register hooks
        hooks = []
        for layer_idx in self.layer_idxs:
            mlp = self.gpt.transformer.h[layer_idx].mlp
            ln2 = self.gpt.transformer.h[layer_idx].ln_2
            
            # run post hook for mlp to capture both inputs and outputs
            #seems to work even without disabling compiler
            @torch.compiler.disable(recursive=False)
            def mlp_hook_fn(module, inputs, outputs, layer_idx=layer_idx):
                # TODO: Why is inputs wrapped in a tuple, but outputs is not?
                # Why don't
                print(outputs)
                activations[f'{layer_idx}_mlpin'] = inputs[0]
                activations[f'{layer_idx}_mlpout'] = outputs[0]
                activations[f'{layer_idx}_mlpactgrads'] = outputs[1]
                print(f"Hook for layer {layer_idx}, got grads")
                # dirty trick: monkey patch the output of the mlp to return the output and act_grads
                # and then hook to capture the grads, and return only the post_mlp output
                return outputs[0] 
            
            # @torch.compiler.disable(recursive=False)
            # def mlp_prehook_fn(module, inputs, layer_idx=layer_idx):
            #      # return the input and a flag to indicate we want the grads
            #     print(f"Prehook for layer {layer_idx}, enable mlp grads")
            #     return (inputs[0], True)
            
            # run pre hook for ln2 to capture resid_mid
            # need to sneak them out of the hook_fn
            
            # If you don't disable compiler, you get an error
            # about 0_residmid not being found???
            @torch.compiler.disable(recursive=False)
            def ln2_hook_fn(module, inputs, layer_idx=layer_idx):
                activations[f'{layer_idx}_residmid'] = inputs[0]

            
            hooks.append(ln2.register_forward_pre_hook(ln2_hook_fn))  # type: ignore
            #hooks.append(mlp.register_forward_pre_hook(mlp_prehook_fn))  # type: ignore
            hooks.append(mlp.register_forward_hook(mlp_hook_fn))  # type: ignore
            
        try:
            yield activations

        finally:
            # Unregister hooks
            for hook_fn in hooks:
                hook_fn.remove()
                
    def post_sae_forward(self, 
                activations: dict[str, torch.Tensor], 
                encoder_outputs: dict[str, EncoderOutput]
    ) -> dict[str, EncoderOutput]:
        """
        Compute and inject the jacobian loss term into the encoder outputs as aux loss
        """
        for layer_idx in self.layer_idxs:
            topk_indices_mlpin = encoder_outputs[f'{layer_idx}_mlpin'].indices
            topk_indices_mlpout = encoder_outputs[f'{layer_idx}_mlpout'].indices
            
            mlp_act_grads = activations[f"{layer_idx}_mlpactgrads"]
            
            jacobian = get_jacobian(
                sae_mlpin = self.saes[f'{layer_idx}_mlpin'],
                sae_mlpout = self.saes[f'{layer_idx}_mlpout'],
                mlp = self.gpt.transformer.h[layer_idx].mlp,
                topk_indices_mlpin = topk_indices_mlpin,
                topk_indices_mlpout = topk_indices_mlpout,
                mlp_act_grads = mlp_act_grads,
            )
            assert self.saes[f'{layer_idx}_mlpin'].k == self.saes[f'{layer_idx}_mlpout'].k, "k values must be the same"
            k = self.saes[f'{layer_idx}_mlpin'].k
            
            j_coeff = self.loss_coefficients.sparsity[layer_idx] 
            jacobian_loss = j_coeff * torch.abs(jacobian).sum() / (k ** 2)
            
            # Each SAE has it's own loss term, and are trained "independently"
            # so we will put the jacobian loss into the aux loss term
            # for the sae_mlpout for each pair of SAEs
            encoder_outputs[f'{layer_idx}_mlpout'].loss.aux = jacobian_loss
            
        return encoder_outputs

