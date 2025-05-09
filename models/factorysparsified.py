"""
Helper class for using the appropriate flavour of SparsifiedGPT model.

Example of use:

device = torch.device("cuda")
gpt_mlp = FactorySparsified.load("checkpoints/jblock.shk_64x4-sparse-2.2e-04", device=device)
"""

from config.sae.models import SAEConfig, SAEVariant
from models.jsaesparsified import JSparsifiedGPT
from models.jsaeblockparsified import JBlockSparsifiedGPT
from models.sparsified import SparsifiedGPT
from models.staircaseblocksparsified import StaircaseBlockSparsifiedGPT
from models.gpt import GPT
from models.sae import SparseAutoencoder
from config.sae.training import LossCoefficients

import os
import json
from pathlib import Path
import torch
from typing import Optional
class FactorySparsified(SparsifiedGPT):

    @classmethod
    def make(cls,
             config: SAEConfig,
             loss_coefficients: Optional[LossCoefficients] = None,
             trainable_layers: Optional[list[int]] = None):
        
        match config.sae_variant:
            case SAEVariant.JSAE:
                return JSparsifiedGPT(config, loss_coefficients, trainable_layers)
            case SAEVariant.JSAE_BLOCK:
                return JBlockSparsifiedGPT(config, loss_coefficients, trainable_layers)
            case SAEVariant.STAIRCASE_BLOCK:
                return StaircaseBlockSparsifiedGPT(config, loss_coefficients, trainable_layers)
            case _:
                return SparsifiedGPT(config, loss_coefficients, trainable_layers)
            
    @classmethod
    def load(cls, dir, loss_coefficients=None, trainable_layers=None, device: torch.device = torch.device("cpu")):
        model = super().load(dir, loss_coefficients, trainable_layers, device)
    
        # Dirty hack while I find the real cause of the bug
        if isinstance(model, StaircaseBlockSparsifiedGPT):
            for key, sae in model.saes.items():
                model.saes[key].shared_context.W_dec.data = sae.shared_context.W_dec.data.to(device)
                model.saes[key].shared_context.W_enc.data = sae.shared_context.W_enc.data.to(device)
                model.saes[key].W_dec.data = sae.W_dec.data.to(device)
                model.saes[key].W_enc.data = sae.W_enc.data.to(device)

        return model