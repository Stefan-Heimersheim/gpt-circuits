import torch
import torch.nn as nn
from config.gpt.models import GPTConfig
from models.gpt import GPT
from models.mlpgpt import MLP_GPT
from safetensors.torch import load_model
import os
import json
from jaxtyping import Float
from torch import Tensor
from typing import Iterable, Optional

from training import Trainer
from config import TrainingConfig
from config.sae.training import LossCoefficients
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput
from models.sae import EncoderOutput

from dataclasses import dataclass

import torch.nn.functional as F

class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias
        
@dataclass
class DyTSparsifiedGPTOutput():
    logits: torch.Tensor
    cross_entropy_loss: torch.Tensor
    activations: dict[str, torch.Tensor]
    ce_loss_increases: Optional[torch.Tensor]
    compound_ce_loss_increase: Optional[torch.Tensor]
    dyt_loss_components: dict[str, torch.Tensor]
    reconstructed_activations: dict[str, torch.Tensor]
        
class DyTSparsifiedGPT(SparsifiedGPT):
    def __init__(self, 
                 config: GPTConfig, 
                 loss_coefficients: Optional[LossCoefficients] = None, 
                 trainable_layers: Optional[tuple] = None):
        nn.Module.__init__(self) 
        self.config = config
        self.loss_coefficients = loss_coefficients
        self.gpt = MLP_GPT(config.gpt_config)
        self.layer_idxs = trainable_layers if trainable_layers else list(range(self.gpt.config.n_layer))
        self.dyts = nn.ModuleDict([(f"{i}", DyT(config.n_embd)) for i in self.layer_idxs])
        
    def forward_with_patched_activations(self, 
                                         ln2out: torch.Tensor, 
                                         resid_mid: torch.Tensor,
                                         layer_idx: int) -> torch.Tensor:
        """
        Forward pass of the model with patched activations.
        """
        return super().forward_with_patched_activations(ln2out, resid_mid, layer_idx, "mlpin")
    
    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, is_eval: bool = False
    ) -> SparsifiedGPTOutput:
        """
        Forward pass of the sparsified model.

        :param idx: Input tensor.
        :param targets: Target tensor.
        :param is_eval: Whether the model is in evaluation mode.
        """
        activations: dict[str, torch.Tensor]
        dyt_outputs: dict[str, torch.Tensor]
        
        with self.record_activations() as activations:
            with self.use_dyts() as encoder_outputs:
                logits, cross_entropy_loss = self.gpt(idx, targets)
        # print(cross_entropy_loss) # Optional: Keep for debugging
        # print(self.resid_mid_cache) # Optional: Keep for debugging
        #torch.cuda.synchronize()
        #print("SLOW DOWN BUDDY")
        
        # If targets are provided during training evaluation, gather more metrics
        ce_loss_increases = None
        compound_ce_loss_increase = None
        if is_eval and targets is not None:
            # Calculate cross-entropy loss increase for each SAE layer
            ce_loss_increases = []
            for layer_idx, dyt in enumerate(self.dyts):
                recon_ln2_out = encoder_outputs[f'{layer_idx}_ln2out']
                resid_mid = activations[f'{layer_idx}_residmid']
                
                dyt_logits = self.gpt.forward_with_patched_activations(recon_ln2_out, resid_mid, layer_idx)
                ce_loss_increases.append(F.cross_entropy(dyt_logits.view(-1, dyt_logits.size(-1)), targets.view(-1)) - cross_entropy_loss)
            ce_loss_increases = torch.stack(ce_loss_increases)

            # Calculate compound cross-entropy loss as a result of patching activations.
            with self.use_dyts(activations_to_patch=self.layer_idxs):
                _, compound_cross_entropy_loss = self.gpt(idx, targets)
                compound_ce_loss_increase = compound_cross_entropy_loss - cross_entropy_loss

        loss = {}
        for i in self.layer_idxs:
            loss[i] = F.mse_loss(dyt_outputs[f'{i}_dyt'], dyt_outputs[f'{i}_ln'])
            
        return DyTSparsifiedGPTOutput(
            logits=logits,
            cross_entropy_loss=cross_entropy_loss,
            activations=activations,
            ce_loss_increases=ce_loss_increases,
            compound_ce_loss_increase=compound_ce_loss_increase,
            dyt_loss_components=loss,
            reconstructed_activations={i: dyt_outputs[f'{i}_dyt'] for i in self.layer_idxs},
        )
        
    @contextmanager
    def record_activations(self):
        """
        Context manager for recording residual stream activations.

        :yield activations: Dictionary of activations.
        activations[f'{layer_idx}_residmid'] = h[layer_idx].ln2in
        activations[f'{layer_idx}_mlpin'] = h[layer_idx].ln2out
        # NOTE: resid_mid is stored in self.resid_mid_cache, not yielded directly
        """
        activations: dict[str, torch.Tensor] = {}

        # Register hooks
        hooks = []
        for layer_idx in self.layer_idxs:
            ln2 = self.gpt.transformer.h[layer_idx].ln_2
         
            @torch.compiler.disable(recursive=False)
            def ln2_hook_fn(module, inputs, output, layer_idx=layer_idx):
                activations[f'{layer_idx}_residmid'] = inputs[0]
                activations[f'{layer_idx}_mlpin'] = output

            hooks.append(ln2.register_forward_hook(ln2_hook_fn))  # type: ignore

        try:
            yield activations

        finally:
            # Unregister hooks
            for hook_fn in hooks:
                hook_fn.remove()
        
    @contextmanager
    def use_dyts(self, activations_to_patch: Iterable[str] = ()):
        """
        Context manager for using SAE layers during the forward pass.

        :param activations_to_patch: Layer indices and hook locations for patching residual stream activations with reconstructions.
        :yield encoder_outputs: Dictionary of encoder outputs.
        key = f"{layer_idx}_{hook_loc}" e.g. 0_mlpin, 0_mlpout, 1_mlpin, 1_mlpout, etc.
        """
        # Dictionary for storing results
        dyt_outputs: dict[str, torch.Tensor] = {}

        # Register hooks
        hooks = []
        for layer_idx, dyt in enumerate(self.dyts):
            target = self.gpt.transformer.h[layer_idx].ln_2
            should_patch_activations = layer_idx in activations_to_patch
            #hook_fn = self.create_sae_hook(sae, encoder_outputs, sae_key, should_patch_activations)
            
            def hook_fn(module, inputs, output, layer_idx=layer_idx):
                ln_approx = self.dyts[layer_idx](inputs[0])
                dyt_outputs[f'{layer_idx}_dyt'] = ln_approx
                dyt_outputs[f'{layer_idx}_ln'] = output
                
                if should_patch_activations:
                    return ln_approx
                else:
                    return output
        
            hooks.append(target.register_forward_hook(hook_fn))  # type: ignore

        try:
            yield dyt_outputs

        finally:
            # Unregister hooks
            for hook_fn in hooks:
                hook_fn.remove()


        
        
class DyTTrainer(Trainer):
    def __init__(self, model: DyTSparsifiedGPT, config: TrainingConfig):
        super().__init__(model, config)
        
    def calculate_loss(self, x, y, is_eval: bool
        ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        output: DyTSparsifiedGPTOutput = self.model(x, y, is_eval=is_eval)
        
        
        


