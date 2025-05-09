import json
import os
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional, Tuple, Type, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

from config.sae.models import SAEConfig, SAEVariant
from config.sae.training import LossCoefficients
from models.gpt import GPT, MLP

from models.sae.topk import TopKSharedContext

from config.sae.models import HookPoint

from config.gpt.models import NormalizationStrategy
from models.sparsified import SparsifiedGPT
from models.sae import EncoderOutput, SparseAutoencoder
from models.sparsified import SparsifiedGPTOutput
import torch.nn.functional as F

from models.jsaesparsified import JSparsifiedGPT

from jaxtyping import Float
from torch import Tensor
from typing import Literal

class JBlockSparsifiedGPT(SparsifiedGPT):
    def __init__(
        self, 
        config: SAEConfig,
        loss_coefficients: Optional[LossCoefficients] = None,
        trainable_layers: Optional[tuple] = None,
    ):
        # TODO: more elegant way to just allow for arbitrary hook points to attach SAEs to,
        # rather than a jillion different SparsifiedGPT subclasses.
        assert config.sae_variant == SAEVariant.JSAE_BLOCK, f"You must use JSAE_BLOCK variant. See JSparsifiedGPT/SparsifiedGPT for other variants."
        
        nn.Module.__init__(self) 
        self.config = config
        self.loss_coefficients = loss_coefficients
        self.gpt = GPT(config.gpt_config)
        
        assert len(config.n_features) == self.gpt.config.n_layer * 2
        self.layer_idxs = trainable_layers if trainable_layers else list(range(self.gpt.config.n_layer))
        sae_keys = [f'{x}_{y}' for x in self.layer_idxs for y in [HookPoint.RESID_MID.value, HookPoint.RESID_POST.value]]  
        
        sae_class: Type[SparseAutoencoder] = self.get_sae_class(config)
        
        self.saes = nn.ModuleDict(dict([(key, sae_class(idx, config, loss_coefficients, self)) 
                                        for idx, key in enumerate(sae_keys)]))
        
    
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
        encoder_outputs: dict[str, EncoderOutput]
        
        with self.record_activations() as activations:
            with self.use_saes() as encoder_outputs:
                logits, cross_entropy_loss = self.gpt(idx, targets)
        
        # If targets are provided during training evaluation, gather more metrics
        ce_loss_increases = None
        compound_ce_loss_increase = None
        if is_eval and targets is not None:
            # Calculate cross-entropy loss increase for each SAE pair
            ce_loss_increases = []
            for layer_idx in self.layer_idxs:
                sae_key = f"{layer_idx}_residmid"
                sae_logits = self.forward_with_sae_pair(layer_idx, encoder_outputs[sae_key].reconstructed_activations)
                sae_ce_loss = F.cross_entropy(sae_logits.view(-1, sae_logits.size(-1)), targets.view(-1))
                ce_loss_increases.append(sae_ce_loss - cross_entropy_loss)
            ce_loss_increases = torch.stack(ce_loss_increases)

            # Calculate compound cross-entropy loss as a result of patching activations.
            with self.use_saes(activations_to_patch=self.saes.keys()):
                _, compound_cross_entropy_loss = self.gpt(idx, targets)
                compound_ce_loss_increase = compound_cross_entropy_loss - cross_entropy_loss

        return SparsifiedGPTOutput(
            logits=logits,
            cross_entropy_loss=cross_entropy_loss,
            activations=activations,
            ce_loss_increases=ce_loss_increases,
            compound_ce_loss_increase=compound_ce_loss_increase,
            sae_loss_components={i: output.loss for i, output in encoder_outputs.items() if output.loss},
            feature_magnitudes={i: output.feature_magnitudes for i, output in encoder_outputs.items()},
            reconstructed_activations={i: output.reconstructed_activations for i, output in encoder_outputs.items()},
            indices={i: output.indices for i, output in encoder_outputs.items()},
        )
        
    # TODO: would be nice to have all the sparsifyGPTs under one class, and then branchout from there?
    def forward_with_sae_pair(self, 
                            layer_idx: int,    
                            resid_mid_recon: Float[Tensor, "B T n_embd"]) -> torch.Tensor:
        """
        Forward pass of the model with patched activations.
        """
        sae_key = f"{layer_idx}_residpost"
        block = self.gpt.transformer.h[layer_idx]
        resid_post = block.mlp(block.ln_2(resid_mid_recon)) + resid_mid_recon
        resid_post_recon = self.saes[sae_key](resid_post).reconstructed_activations
        
        return self.gpt.forward(resid_post_recon, start_at_layer=layer_idx+1).logits
            
    
    @contextmanager
    def record_activations(self):
        """
        Context manager for recording residual stream activations.
        """
        act: dict[str, torch.Tensor] = {}

        # Register hooks
        hooks = []
        for layer_idx in self.layer_idxs:
            block = self.gpt.transformer.h[layer_idx]
            
            self.make_cache_post_hook(hooks, act, block, key_out = f"{layer_idx}_residpost")
            
            if "jsae" in self.config.sae_variant:       
                self.make_grad_hook(hooks, act, block.mlp.gelu, key = f"{layer_idx}_mlpactgrads")
            # Adding two hooks is okay, will execute in order
            self.make_cache_pre_hook(hooks, act, block.ln_2, key_in = f"{layer_idx}_residmid") 
            
            if "jsae" in self.config.sae_variant:
                self.make_grad_hook(hooks, act, block.ln_2, key = f"{layer_idx}_normactgrads")
    
        try:
            yield act

        finally:
            # Unregister hooks
            for hook_fn in hooks:
                hook_fn.remove()
                
    @contextmanager
    def use_saes(self, activations_to_patch: Iterable[str] = ()):
        """
        Context manager for using SAE layers during the forward pass.

        :param activations_to_patch: Layer indices and hook locations for patching residual stream activations with reconstructions.
        :yield encoder_outputs: Dictionary of encoder outputs.
        key = f"{layer_idx}_{hook_loc}" e.g. 0_mlpin, 0_mlpout, 1_mlpin, 1_mlpout, etc.
        """
        encoder_outputs: dict[str, EncoderOutput] = {}

        hooks = []
        for layer_idx in self.layer_idxs:

            block = self.gpt.transformer.h[layer_idx]

            sae_key = f'{layer_idx}_residmid'
            self.make_sae_pre_hook(hooks, encoder_outputs, block.ln_2, sae_key, activations_to_patch)
            
            sae_key = f'{layer_idx}_residpost'
            self.make_sae_post_hook(hooks, encoder_outputs, block, sae_key, activations_to_patch)
            
        try:
            yield encoder_outputs

        finally:
            for hook_fn in hooks:
                hook_fn.remove()