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

class BlockSparsifiedGPT(StaircaseBlockSparsifiedGPT):
    def __init__(
        self, 
        config: SAEConfig,
        loss_coefficients: Optional[LossCoefficients] = None,
        trainable_layers: Optional[tuple] = None,
    ):
        # TODO: more elegant way to just allow for arbitrary hook points to attach SAEs to,
        # rather than a jillion different SparsifiedGPT subclasses.
        #assert config.sae_variant == SAEVariant.STAIRCASE_BLOCK, f"You must use STAIRCASE_BLOCK variant. See JSparsifiedGPT/SparsifiedGPT for other variants."
        
        nn.Module.__init__(self) 
        self.config = config
        self.loss_coefficients = loss_coefficients
        self.gpt = GPT(config.gpt_config)
        
        assert len(config.n_features) == self.gpt.config.n_layer * 2
        self.layer_idxs = trainable_layers if trainable_layers else list(range(self.gpt.config.n_layer))
        
        self.sae_keys = [f'{x}_{y}' for x in self.layer_idxs for y in (HookPoint.RESID_MID.value, HookPoint.RESID_POST.value)]

        sae_class: Type[SparseAutoencoder] = self.get_sae_class(config)
        
        self.saes = nn.ModuleDict(dict([(key, sae_class(idx, config, loss_coefficients, self)) 
                                        for idx, key in enumerate(self.sae_keys)]))