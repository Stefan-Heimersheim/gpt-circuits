
import torch.nn as nn
import torch
from typing import Optional, Type
from pathlib import Path

from safetensors.torch import save_model, load_model

from models.sae import SparseAutoencoder
from config.sae.models import SAEConfig
from config.sae.training import LossCoefficients


class StaircaseBaseSAE():
    """
    TopKSAEs that share weights between layers, and each child uses slices into weights inside shared context.
    """
        
    def save(self, dirpath: Path) -> None:
        # Save non-shared parameters
        child_path = dirpath / f"sae.{self.layer_idx}.safetensors"
        non_shared_params = {name: param for name, param in self.named_parameters() if not name.startswith('shared_context')}
        tmp_module = nn.ParameterDict(non_shared_params)
        save_model(tmp_module, str(child_path))

        # Save shared parameters
        if self.is_first:
            shared_path = dirpath / f"sae.shared.{self.shared_context.layer_idx}.safetensors"
            print(f"Saving shared parameters to {shared_path}")
            save_model(self.shared_context, str(shared_path))

    def load(self, dirpath: Path, device: torch.device):
        # Load non-shared parameters
        child_path = dirpath / f"sae.{self.layer_idx}.safetensors"
        non_shared_params = {name: torch.empty_like(param) for name, param in self.named_parameters() if not name.startswith('shared_context')}
        tmp_module = nn.ParameterDict(non_shared_params)
        load_model(tmp_module, str(child_path), device=device.type)

        for name, param in self.named_parameters():
            if not name.startswith('shared_context'):
                param.data = tmp_module[name]

        # Load shared parameters
        if hasattr(self, 'is_first') and self.is_first and hasattr(self, 'shared_context'):
            # Try loading from layer_idx specific file first
            shared_path = dirpath / f"sae.shared.{self.shared_context.layer_idx}.safetensors"
            if shared_path.exists():
                print(f"Loading shared parameters from {shared_path}")
                load_model(self.shared_context, shared_path, device=device.type)
            elif self.shared_context.layer_idx == 0:
                # Fallback to generic shared file for layer_idx=0
                fallback_path = dirpath / "sae.shared.safetensors"
                if fallback_path.exists():
                    import warnings
                    warnings.warn(f"DEPRICATED: Using fallback shared parameters path: {fallback_path}")
                    load_model(self.shared_context, fallback_path, device=device.type)
                else:
                    raise FileNotFoundError(f"Could not find shared parameters at {shared_path} or fallback at {fallback_path}")
            else:
                raise FileNotFoundError(f"Could not find shared parameters at {shared_path}")
