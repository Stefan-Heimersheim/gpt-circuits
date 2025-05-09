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
from models.gpt import GPT
from models.sae import SparseAutoencoder
from config.sae.training import LossCoefficients

import os
import json
from pathlib import Path
import torch
from typing import Optional
class FactorySparsified():

    @classmethod
    def make(cls,
             config: SAEConfig,
             loss_coefficients: Optional[LossCoefficients] = None,
             trainable_layers: Optional[list[int]] = None):
        if config.sae_variant == SAEVariant.JSAE:
            return JSparsifiedGPT(config, loss_coefficients, trainable_layers)
        elif config.sae_variant == SAEVariant.JSAE_BLOCK:
            return JBlockSparsifiedGPT(config, loss_coefficients, trainable_layers)
        else:
            return SparsifiedGPT(config, loss_coefficients, trainable_layers)
        
    @classmethod
    def load(cls, dir, loss_coefficients=None, trainable_layers=None, device: torch.device = torch.device("cpu")):
        """
        Load a sparsified GPT model from a directory.
        """
        # Load GPT model
        gpt = GPT.load(dir, device=device)

        # Load SAE config
        meta_path = os.path.join(dir, "sae.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        config = SAEConfig(**meta)
        config.gpt_config = gpt.config

        # Create model using saved config
        print(cls)
        model = FactorySparsified.make(config, loss_coefficients, trainable_layers)
        model.gpt = gpt

        # Load SAE weights
        for module in model.saes.values():
            assert isinstance(module, SparseAutoencoder)
            module.load(Path(dir), device=device)

        return model
            
            