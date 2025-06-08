"""
Train SAE weights for all layers concurrently.

$ python -m training.sae.staircase_concurrent --config=topk-staircase-share.shakespeare_64x4 [--load_from=shakespeare_64x4]
$ torchrun --standalone --nproc_per_node=8 -m training.sae.staircase_concurrent --config=topk-staircase-share.stories_256x4 [--load_from=stories_256x4]
"""

from pathlib import Path

import torch

from models.sparsified import SparsifiedGPT
from training.sae.concurrent import ConcurrentTrainer
from models.sparsified import SparsifiedGPTOutput

import einops

from torch import Tensor
from jaxtyping import Bool, Float
from typing import Tuple


class StaircaseConcurrentTrainer(ConcurrentTrainer):
    """
    Train SAE weights for all layers concurrently.
    """

    def save_checkpoint(self, model: SparsifiedGPT, is_best: torch.Tensor):
        """
        Save SAE weights for layers that have achieved a better validation loss.
        """
        # As weights are shared, only save if each layer is better than the previous best.
        
        if torch.all(is_best):
            layers_to_save = list(self.model.saes.keys())
        else:
            layers_to_save = None
            
        model.save(self.config.out_dir, layers_to_save)
        
        # is_best : Bool[Tensor, "2 * n_layer"]
        # is_best = einops.rearrange(is_best, "(n_layer loc) -> n_layer loc", loc=2)
        # is_best = torch.all(is_best, dim=-1) # (n_layer)
        
        # for layer_idx in self.model.layer_idxs:
        #     if is_best[layer_idx]:
        #         for loc in [HookPoint.RESID_MID.value, HookPoint.RESID_POST.value]:
        #             sae_key = f"{layer_idx}_{loc}"
        #             sae = self.model.saes[sae_key]
        #             assert "staircase" in sae.config.sae_variant, \
        #                 f"Staircase trainer must use staircase SAE variant, Error: {sae.config.sae_variant}"
        #             sae.save(Path(dir))
                
    def gather_metrics(self, loss: torch.Tensor, output: SparsifiedGPTOutput) -> dict[str, torch.Tensor]:
        """
        Gather metrics from loss and model output.
        """
        metrics = super().gather_metrics(loss, output)
        del metrics["l0s"] # don't need l0s per layer as for topk sae, l0 = k by definition
        
        l0_per_chunk = {}
        for sae_idx, (sae_key, feature_magnitudes) in enumerate(output.feature_magnitudes.items()):
            #n_features = tuple(8,16,24,32,40) for 4 layer network with 5 chunks
            features = torch.tensor(self.config.sae_config.n_features[:sae_idx+1], 
                                    dtype=torch.int32)
            zero = torch.tensor([0], dtype=torch.int32)
            chunk_sizes = torch.cat([zero, features])
            splits = torch.diff(chunk_sizes)
            
            # Slice into chunks: f[0:8], f[8:16], ...
            chunks : Tuple[Float[Tensor, "batch seq chunk_size"], ...]
            chunks = torch.split(feature_magnitudes, splits.tolist(), dim=-1)
            
            l0_per_chunk = []
            # Calculate L0 for each chunk
            for chunk in chunks:
                avg_l0 = (chunk != 0).float().sum(dim=-1).mean()
                l0_per_chunk.append(avg_l0)
            l0_per_chunk = torch.stack(l0_per_chunk)
            metrics[f"l0_{sae_key}"] = l0_per_chunk
        
        return metrics
