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
from models.sae.topk import TopKSharedContext, StaircaseTopKSAE
from config.sae.training import LossCoefficients
from models.gpt import GPT, MLP

from config.sae.models import HookPoint

from config.gpt.models import NormalizationStrategy
from models.sparsified import SparsifiedGPT
from models.sae import EncoderOutput, SparseAutoencoder
from models.sparsified import SparsifiedGPTOutput
from models.jsaeblockparsified import JBlockSparsifiedGPT
import torch.nn.functional as F

from models.jsaesparsified import JSparsifiedGPT

from jaxtyping import Float
from torch import Tensor
from typing import Literal

class StaircaseBlockSparsifiedGPT(JBlockSparsifiedGPT):
    def __init__(
        self, 
        config: SAEConfig,
        loss_coefficients: Optional[LossCoefficients] = None,
        trainable_layers: Optional[tuple] = None,
    ):
        # TODO: more elegant way to just allow for arbitrary hook points to attach SAEs to,
        # rather than a jillion different SparsifiedGPT subclasses.
        assert config.sae_variant == SAEVariant.STAIRCASE_BLOCK, f"You must use STAIRCASE_BLOCK variant. See JSparsifiedGPT/SparsifiedGPT for other variants."
        
        nn.Module.__init__(self) 
        self.config = config
        self.loss_coefficients = loss_coefficients
        self.gpt = GPT(config.gpt_config)
        
        assert len(config.n_features) == self.gpt.config.n_layer * 2, f"StaircaseBlockSparsifiedGPT: n_features must be twice the number of layers. Got {len(config.n_features)} and {self.gpt.config.n_layer}."
        self.layer_idxs = trainable_layers if trainable_layers else list(range(self.gpt.config.n_layer))
        
        self.sae_keys = [f'{x}_{y}' for x in self.layer_idxs for y in (HookPoint.RESID_MID.value, HookPoint.RESID_POST.value)]

        self.saes = {}
        self.shared_context = {}
        for i in self.layer_idxs:
            feature_size = config.n_features[2*i+1]
            parent = TopKSharedContext(i, feature_size, config)
            self.shared_context[i] = parent
            self.saes[f'{i}_{HookPoint.RESID_POST.value}'] = StaircaseTopKSAE(2*i+1, config, loss_coefficients, parent, is_first = True)
            self.saes[f'{i}_{HookPoint.RESID_MID.value}'] = StaircaseTopKSAE(2*i, config, loss_coefficients, parent, is_first = False)
                
                
        self.saes = nn.ModuleDict(self.saes)
    
    def __repr__(self):
        """
        Custom __repr__ to avoid circular reference with shared_context.
        """
        return f"StaircaseBlockSparsifiedGPT(\n  (gpt): {self.gpt}\n  (saes): {self.saes}\n  (shared_context): <{len(self.shared_context)} shared contexts>\n)"
    
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
            self.make_cache_pre_hook(hooks, act, block.ln_2, key_in = f"{layer_idx}_residmid") 
    
        try:
            yield act

        finally:
            # Unregister hooks
            for hook_fn in hooks:
                hook_fn.remove()