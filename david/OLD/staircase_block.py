
import torch.nn as nn
import torch
from typing import Optional, Type
from pathlib import Path

from safetensors.torch import save_model, load_model

from models.sae import SparseAutoencoder
from config.sae.models import SAEConfig, HookPoint
from config.sae.training import LossCoefficients
from models.sae.topk import TopKSharedContext

class StaircaseFactory():
    def __init__(self, config: SAEConfig, loss_coefficients: Optional[LossCoefficients]):
        self.layer_idxs = config.layer_idxs
        self.hook_points = config.hook_points
        self.sae_keys = [f'{x}_{y}' for x in self.layer_idxs for y in self.hook_points]

        saes = {}

        for layer_idx in self.layer_idxs:
            parent_sae = TopKSharedContext(layer_idx, config, loss_coefficients)
            for hook_point in self.hook_points:
                child_sae = StaircaseBlockSAE(layer_idx, config, loss_coefficients, parent_sae)


class StaircaseBlockBaseSAE(nn.Module):
    """
    A base class for other SAEs to share weights from.
    Allows for multiple independant staircase SAEs to be attached to the same model.
    """

    def __init__(self, layer_idx: int, config: SAEConfig, loss_coefficients: Optional[LossCoefficients]):

        # Shared context from which we can get weight parameters.
        assert "staircase" in config.sae_variant, "staircase variant must be used with staircase SAEs"
        self.is_first = False
        if not hasattr(model, "shared_context"):
            # Initialize the shared context once.
            self.is_first = True
            model.shared_context = shared_context(config)
        self.shared_context = model.shared_context  # type: ignore
        
    def save(self, dirpath: Path) -> None:
        # Save non-shared parameters
        child_path = dirpath / f"sae.{self.layer_idx}.safetensors"
        non_shared_params = {name: param for name, param in self.named_parameters() if not name.startswith('shared_context')}
        tmp_module = nn.ParameterDict(non_shared_params)
        save_model(tmp_module, str(child_path))

        # Save shared parameters
        if self.is_first:
            shared_path = dirpath / "sae.shared.safetensors"
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
        if self.is_first:
            shared_path = dirpath / "sae.shared.safetensors"
            load_model(self.shared_context, shared_path, device=device.type)
