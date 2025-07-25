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
from config.sae.models import SAEVariant, HookPoint
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
        # As weights are shared, only save if all layers are better.
        if torch.all(is_best):
            model.save(self.config.out_dir)
        
    @torch.no_grad()
    def gather_metrics(self, loss: torch.Tensor, output: SparsifiedGPTOutput) -> dict[str, torch.Tensor]:
        """
        Gather metrics from loss and model output.
        """
        metrics = super().gather_metrics(loss, output)
        #del metrics["l0s"] # don't need l0s per layer as for topk sae, l0 = k by definition
        
        l0_per_chunk = {}
        
        if self.config.sae_config.sae_variant == SAEVariant.TOPK_STAIRCASE:
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
                
        elif self.config.sae_config.sae_variant == SAEVariant.STAIRCASE_BLOCK:
            # TODO: this is trash but I just wanted to get something working fast. oops. -David
            pass

            # for sae_idx, (sae_key, feature_magnitudes) in enumerate(output.feature_magnitudes.items()):
            #     layer_idx, loc = sae_key.split("_")
            #     layer_idx = int(layer_idx)
                
            #     if loc == HookPoint.RESID_MID.value:
            #         avg_l0 = (feature_magnitudes != 0).float().sum(dim=-1).mean()
            #         metrics[f"l0_{sae_key}"] = [avg_l0]
                    
            #     elif loc == HookPoint.RESID_POST.value:
   
            #         # just cut the damned thing in half. if you wanted a growth factor that is not 2, too bad!
            #         half = feature_magnitudes.shape[-1] // 2
            #         lower_chunk, upper_chunk = torch.split(feature_magnitudes, [half, half], dim=-1)
            #         avg_l0_lower = (lower_chunk != 0).float().sum(dim=-1).mean()
            #         avg_l0_upper = (upper_chunk != 0).float().sum(dim=-1).mean() 
            #         metrics[f"l0_{sae_key}"] = [avg_l0_lower, avg_l0_upper]
            #     else:
            #         raise ValueError(f"StaircaseConcurrentTrainer: gather_metrics: Invalid location: {loc}")
                
                

        
        return metrics
