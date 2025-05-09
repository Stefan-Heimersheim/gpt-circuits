import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

from config.sae.models import SAEConfig
from config.sae.training import LossCoefficients
from models.gpt import GPT
from models.sae import EncoderOutput, SparseAutoencoder
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput

from typing import Literal

from jaxtyping import Float
from torch import Tensor

# Depricated
class MLPSparsifiedGPT(SparsifiedGPT):
    def __init__(
        self, 
        config: SAEConfig,
        loss_coefficients: Optional[LossCoefficients] = None,
        trainable_layers: Optional[tuple] = None,
    ):
        #don't actually want to call SparsifiedGPT.__init__, but we want to inherit from it
        super().__init__(config, loss_coefficients, trainable_layers)
                
    # def post_init(self):
    #     pass
        # While a nice idea, it might break other code as TopKSAE
        # has a different save format 
        # for sae_key, sae in self.saes.items():
        #     self.saes[sae_key].sae_key = sae_key
        
        # def sae_save(self, dirpath: Path):
        #     """
        #     Save the sparse autoencoder to a file in the specified directory.
        #     """
        #     sae_key = self.sae_key
        #     weights_path = dirpath / f"sae.{sae_key}.safetensors"
        #     save_model(self, str(weights_path))

        # def sae_load(self, dirpath: Path, device: torch.device):
        #     """
        #     Load the sparse autoencoder from a file in the specified directory.
        #     """
        #     sae_key = self.sae_key
        #     weights_path = dirpath / f"sae.{sae_key}.safetensors"
        #     load_model(self, weights_path, device=device.type)
        
        # for sae_key, sae in self.saes.items():
        #     sae.save = lambda dirpath, current_sae=sae: sae_save(current_sae, dirpath)
        #     sae.load = lambda dirpath, device, current_sae=sae: sae_load(current_sae, dirpath, device)
        