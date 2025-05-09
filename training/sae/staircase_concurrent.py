"""
Train SAE weights for all layers concurrently.

$ python -m training.sae.staircase_concurrent --config=topk-staircase-share.shakespeare_64x4 [--load_from=shakespeare_64x4]
$ torchrun --standalone --nproc_per_node=8 -m training.sae.staircase_concurrent --config=topk-staircase-share.stories_256x4 [--load_from=stories_256x4]
"""

import argparse
import dataclasses
import json
import os
from pathlib import Path

import torch

from config import TrainingConfig
from config.sae.models import SAEConfig
from config.sae.training import options
from models.sparsified import SparsifiedGPT
from models.sae.topk import StaircaseTopKSAE
from training.sae.concurrent import ConcurrentTrainer
from models.sparsified import SparsifiedGPTOutput
from models.factorysparsified import FactorySparsified
from config.sae.models import HookPoint

import einops

from torch import Tensor
from jaxtyping import Bool

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Training config")
    parser.add_argument("--load_from", type=str, help="GPT model weights to load", default="shakespeare_64x4")
    parser.add_argument("--name", type=str, help="Model name for checkpoints")
    parser.add_argument("--max_steps", type=int, help="Maximum number of steps to train")
    return parser.parse_args()


class StaircaseConcurrentTrainer(ConcurrentTrainer):
    """
    Train SAE weights for all layers concurrently.
    """

    def save_checkpoint(self, model: SparsifiedGPT, is_best: torch.Tensor):
        """
        Save SAE weights for layers that have achieved a better validation loss.
        """
        # As weights are shared, only save if each layer is better than the previous best.
        dir = self.config.out_dir
        
        self.model.gpt.save(dir)

        # Save SAE config
        meta_path = os.path.join(dir, "sae.json")
        meta = dataclasses.asdict(self.config.sae_config, dict_factory=SAEConfig.dict_factory)
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        
        is_best : Bool[Tensor, "2 * n_layer"]
        is_best = einops.rearrange(is_best, "(n_layer loc) -> n_layer loc", loc=2)
        is_best = torch.all(is_best, dim=-1) # (n_layer)
        
        for layer_idx in self.model.layer_idxs:
            if is_best[layer_idx]:
                for loc in [HookPoint.RESID_MID.value, HookPoint.RESID_POST.value]:
                    sae_key = f"{layer_idx}_{loc}"
                    sae = self.model.saes[sae_key]
                    assert "staircase" in sae.config.sae_variant, \
                        f"Staircase trainer must use staircase SAE variant, Error: {sae.config.sae_variant}"
                    sae.save(Path(dir))
                
    def gather_metrics(self, loss: torch.Tensor, output: SparsifiedGPTOutput) -> dict[str, torch.Tensor]:
        """
        Gather metrics from loss and model output.
        """
        metrics = super().gather_metrics(loss, output)
        
        l0_per_chunk = {}
        for sae_idx, (sae_key, feature_magnitudes) in enumerate(output.feature_magnitudes.items()):
            
            batch, seq, feature_size = feature_magnitudes.shape
            assert feature_size % min(self.model.config.n_features) == 0, f"Feature size {feature_size} must be divisible by n_features {min(self.model.config.n_features)}"
            num_chunks = feature_size // min(self.model.config.n_features) # assume all feature sizes are multiples of the smallest feature size
            
            grouped_feature_magnitudes = torch.chunk(feature_magnitudes, num_chunks, dim=-1) # tuple[(batch, seq, feature_size_each_chunk)]
            grouped_feature_magnitudes = torch.stack(grouped_feature_magnitudes, dim=-2) # (batch, seq, n_chunks, feature_size_each_chunk)
            grouped_l0 = (grouped_feature_magnitudes != 0).float().sum(dim=-1) # (batch, seq, n_chunks)
            l0_per_chunk[sae_idx] = grouped_l0.mean(dim=(0,1)) # (n_chunks)
            metrics[f"l0_{sae_key}"] = l0_per_chunk[sae_idx]
        
        return metrics



    # @classmethod
    # def load(cls, dir, loss_coefficients=None, trainable_layers=None, device: torch.device = torch.device("cpu")):
    #     """
    #     Load a sparsified GPT model from a directory.
    #     """
    #     # Load GPT model
    #     gpt = GPT.load(dir, device=device)

    #     # Load SAE config
    #     meta_path = os.path.join(dir, "sae.json")
    #     with open(meta_path, "r") as f:
    #         meta = json.load(f)
    #     config = SAEConfig(**meta)
    #     config.gpt_config = gpt.config

    #     # Create model using saved config
    #     print(cls)
    #     model = cls(config, loss_coefficients, trainable_layers)
    #     model.gpt = gpt

    #     # Load SAE weights
    #     for module in model.saes.values():
    #         assert isinstance(module, SparseAutoencoder)
    #         module.load(Path(dir), device=device)

    #     return model

    # def save(self, dir, layers_to_save: Optional[list[str]] = None):
    #     """
    #     Save the sparsified GPT model to a directory.

    #     :param dir: Directory for saving weights.
    #     :param layers_to_save: Module names for SAE layers to save. If None, all layers will be saved.
    #     """
    #     # Save GPT model
    #     self.gpt.save(dir)

    #     # Save SAE config
    #     meta_path = os.path.join(dir, "sae.json")
    #     meta = dataclasses.asdict(self.config, dict_factory=SAEConfig.dict_factory)
    #     with open(meta_path, "w") as f:
    #         json.dump(meta, f)

    #     # Which layers should we save?
    #     layers_to_save = layers_to_save or list(self.saes.keys())

    #     # Save SAE modules
    #     for layer_name, module in self.saes.items():
    #         if layer_name in layers_to_save:
    #             assert isinstance(module, SparseAutoencoder)
    #             module.save(Path(dir))


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config_name = args.config
    config = options[config_name]
    assert "staircase" in config.sae_config.sae_variant, "Staircase trainer must use staircase SAE variant"
    # Update outdir
    if args.name:
        config.name = args.name
    if args.max_steps:
        config.max_steps = args.max_steps

    # Initialize trainer
    trainer = StaircaseConcurrentTrainer(config, load_from=TrainingConfig.checkpoints_dir / args.load_from)
    trainer.train()
