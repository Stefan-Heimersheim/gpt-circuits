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
from models.gpt import MLP
from models.mlpsparsified import MLPSparsifiedGPT
from models.sae import EncoderOutput, SparseAutoencoder
from models.sparsified import SparsifiedGPTOutput
import torch.nn.functional as F

import warnings

from jaxtyping import Float
from torch import Tensor
from typing import Callable

class JSparsifiedGPT(MLPSparsifiedGPT):
    def __init__(
        self, 
        config: SAEConfig,
        loss_coefficients: Optional[LossCoefficients] = None,
        trainable_layers: Optional[tuple] = None,
    ):
        super().__init__(config, loss_coefficients, trainable_layers)
        if config.sae_variant != SAEVariant.JSAE:
            warnings.warn("JSparsifiedGPT: You must use JSAE variant. See JBlockSparsifiedGPT/SparsifiedGPT for other variants.")
    
    @property
    def eval_keys(self) -> Union[list[str], list[int]]:
        return self.layer_idxs
    
    def get_sae_logits(self, 
                       layer_idx: int, 
                       activations: dict[int, torch.Tensor], 
                       encoder_outputs: dict[int, EncoderOutput]) -> torch.Tensor:
        
        recon_pre_mlp = encoder_outputs[f'{layer_idx}_mlpin'].reconstructed_activations
        resid_mid = activations[f'{layer_idx}_residmid']

        assert isinstance(recon_pre_mlp, torch.Tensor), f"recon_pre_mlp: {recon_pre_mlp}"
        assert isinstance(resid_mid, torch.Tensor), f"resid_mid: {resid_mid}"
        
        post_mlp = self.gpt.transformer.h[layer_idx].mlp(recon_pre_mlp)
        post_mlp_recon = self.saes[f'{layer_idx}_mlpout'](post_mlp).reconstructed_activations
        
        resid_post = post_mlp_recon + resid_mid
        
        return self.gpt.forward(resid_post, start_at_layer=layer_idx+1).logits        
    
    @contextmanager
    def record_activations(self):
        """
        Context manager for recording residual stream activations.

        :yield activations: Dictionary of activations.
        activations[f'{layer_idx}_mlpin'] = h[layer_idx].mlpin
        activations[f'{layer_idx}_mlpout'] = h[layer_idx].mlpout
        """
        act: dict[str, torch.Tensor] = {}

        # Register hooks
        hooks = []
        for layer_idx in self.layer_idxs:
            mlp = self.gpt.transformer.h[layer_idx].mlp
            ln2 = self.gpt.transformer.h[layer_idx].ln_2
            mlp_act_fn = self.gpt.transformer.h[layer_idx].mlp.gelu
            
            self.make_cache_post_hook(hooks, act, mlp, key_in = f"{layer_idx}_mlpin", 
                                                        key_out = f"{layer_idx}_mlpout")
            self.make_cache_pre_hook(hooks, act, ln2, key_in = f"{layer_idx}_residmid")        
            self.make_grad_hook(hooks, act, mlp_act_fn, key = f"{layer_idx}_mlpactgrads")
    
        try:
            yield act

        finally:
            # Unregister hooks
            for hook_fn in hooks:
                hook_fn.remove()