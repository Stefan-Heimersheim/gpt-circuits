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

import warnings

class JBlockSparsifiedGPT(SparsifiedGPT):
    def __init__(
        self, 
        config: SAEConfig,
        loss_coefficients: Optional[LossCoefficients] = None,
        trainable_layers: Optional[tuple] = None,
    ):
        # TODO: more elegant way to just allow for arbitrary hook points to attach SAEs to,
        # rather than a jillion different SparsifiedGPT subclasses.
        if config.sae_variant != SAEVariant.JSAE_BLOCK: 
            warnings.warn(f"JBlockSparsifiedGPT: You must use JSAE_BLOCK variant. See JSparsifiedGPT/SparsifiedGPT for other variants.")
        
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
        
        self.sae_keys = sae_keys
        self.norm_strategy = config.gpt_config.norm_strategy
        
    @property
    def eval_keys(self) -> list[str]:
        return self.layer_idxs
    
    @torch.no_grad()
    def get_sae_logits(self,
                       eval_key : str, 
                       activations: dict[str, torch.Tensor], 
                       encoder_outputs: dict[str, EncoderOutput]) -> torch.Tensor:
        
        assert isinstance(eval_key, int), "eval_key must be an integer for JBlockSparsifiedGPT"
        
        block = self.gpt.transformer.h[eval_key]
        resid_mid_recon = encoder_outputs[f"{eval_key}_{HookPoint.RESID_MID.value}"].reconstructed_activations
        resid_post = block.mlp(block.ln_2(resid_mid_recon)) + resid_mid_recon
        resid_post_recon = self.saes[f"{eval_key}_{HookPoint.RESID_POST.value}"](resid_post).reconstructed_activations
        
        return self.gpt.forward(resid_post_recon, start_at_layer=eval_key+1).logits
    
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
            
            self.make_cache_post_hook(hooks, act, block, key_out = f"{layer_idx}_{HookPoint.RESID_POST.value}")
            
            if "jsae" in self.config.sae_variant:       
                self.make_grad_hook(hooks, act, block.mlp.gelu, key = f"{layer_idx}_{HookPoint.MLP_ACT_GRAD.value}")
            # Adding two hooks is okay, will execute in order
            self.make_cache_pre_hook(hooks, act, block.ln_2, key_in = f"{layer_idx}_{HookPoint.RESID_MID.value}") 
            
            if "jsae" in self.config.sae_variant:
                if self.norm_strategy == NormalizationStrategy.DYNAMIC_TANH:
                    self.make_grad_hook(hooks, act, block.ln_2, key = f"{layer_idx}_{HookPoint.DYT_ACT_GRAD.value}")
                
                elif self.norm_strategy == NormalizationStrategy.LAYER_NORM:
                    pass #already cached residmid
                    
                else:
                    raise ValueError(f"Invalid norm strategy: {self.norm_strategy}")
    
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

            sae_key = f'{layer_idx}_{HookPoint.RESID_MID.value}'
            self.make_sae_pre_hook(hooks, encoder_outputs, block.ln_2, sae_key, activations_to_patch)
            
            sae_key = f'{layer_idx}_{HookPoint.RESID_POST.value}'
            self.make_sae_post_hook(hooks, encoder_outputs, block, sae_key, activations_to_patch)
            
        try:
            yield encoder_outputs

        finally:
            for hook_fn in hooks:
                hook_fn.remove()