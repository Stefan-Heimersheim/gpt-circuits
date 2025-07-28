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

import warnings

class StaircaseSparsifiedGPT(SparsifiedGPT):
    def __init__(
        self, 
        config: SAEConfig,
        loss_coefficients: Optional[LossCoefficients] = None,
        trainable_layers: Optional[tuple] = None,
    ):
        # TODO: more elegant way to just allow for arbitrary hook points to attach SAEs to,
        # rather than a jillion different SparsifiedGPT subclasses.
        assert config.sae_variant == SAEVariant.TOPK_STAIRCASE, f"You must use TOPK_STAIRCASE variant. See JSparsifiedGPT/SparsifiedGPT for other variants."
        
        nn.Module.__init__(self) 
        self.config = config
        self.loss_coefficients = loss_coefficients
        self.gpt = GPT(config.gpt_config)
        warnings.warn("StaircaseSparsifiedGPT: Use of self.saes[i] is deprecated. Use self.saes[f'{i}_{HookPoint.ACT.value}'] instead.")
        assert len(config.n_features) == self.gpt.config.n_layer + 1, f"StaircaseSparsifiedGPT: n_features must be one more than the number of layers. Got {len(config.n_features)} and {self.gpt.config.n_layer}."
        self.layer_idxs = trainable_layers if trainable_layers else list(range(len(config.n_features)))

        self.saes = {}
        n_features_list = list(config.n_features)
        assert n_features_list == sorted(n_features_list), f"StaircaseSparsifiedGPT: n_features must be sorted. Got {n_features_list}."
        self.shared_context = TopKSharedContext(0, config.n_features[-1], config)
        for i in self.layer_idxs:
            is_biggest = i == self.layer_idxs[-1]
            self.saes[f'{i}_{HookPoint.ACT.value}'] = StaircaseTopKSAE(i, config, loss_coefficients, self.shared_context, is_first = is_biggest)
            # Alias i to i_act
            #self.saes[f'{i}'] = self.saes[f'{i}_{HookPoint.ACT.value}']
        self.sae_keys = self.saes.keys()
                
        self.saes = nn.ModuleDict(self.saes)
    
    def __repr__(self):
        """
        Custom __repr__ to avoid circular reference with shared_context.
        """
        return f"StaircaseSparsifiedGPT(\n  (gpt): {self.gpt}\n  (saes): {self.saes}\n  (shared_context): <1 shared context>\n)"
        
