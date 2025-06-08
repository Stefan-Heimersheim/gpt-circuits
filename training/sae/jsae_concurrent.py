"""
python -m training.sae.jsae_concurrent [--config=jsae.shakespeare_64x4] [--load_from=shakespeare_64x4] [--sparsity=0.02|0.1,0.2,0.3,0.4]
"""
# %%
from pathlib import Path

import einops
import torch

from config.sae.training import SAETrainingConfig
from models.jsaesparsified import JSparsifiedGPT
from models.sparsified import SparsifiedGPTOutput
from training.sae.concurrent import ConcurrentTrainer

import warnings
from typing import Optional

# Change current working directory to parent
# while not os.getcwd().endswith("gpt-circuits"):
#     os.chdir("..")
# print(os.getcwd())

from utils.jsae import jacobian_mlp


class JSaeTrainer(ConcurrentTrainer):
    """
    Train SAE weights for all layers concurrently.
    """
            
    def save_checkpoint(self, model: JSparsifiedGPT, is_best: torch.Tensor, locs = ('mlpin', 'mlpout')):
        """
        Save SAE weights for layers that have achieved a better validation loss.
        """
        # As weights are shared, only save if each layer is better than the previous best.
        self.model.gpt.save(self.config.out_dir)
        
        for layer_idx in self.model.layer_idxs:
            if is_best[layer_idx]:
                for loc in locs:
                    sae_key = f'{layer_idx}_{loc}'
                    self.model.saes[sae_key].save(self.config.out_dir)

    # TODO: This is a very expensive operation, we should try to speed it up
    def output_to_loss(self, output: SparsifiedGPTOutput, is_eval: bool= False) -> torch.Tensor:
        """
        Return an array of losses, one for each pair of layers:
        loss[i] = loss_recon[f"{i}_mlpin"] + loss_recon[f"{i}_mlpout"] 
            + l1_jacobian(f"{i}_mlpout", f"{i}_mlpin")
        """
        device = output.sae_losses.device
        recon_losses = output.sae_losses
        jacobian_losses = torch.zeros(len(self.model.layer_idxs), device=device)
        j_coeffs = torch.tensor(self.model.loss_coefficients.sparsity, device=device)
        
        
        for layer_idx in self.model.layer_idxs:
            key = self.model.sae_keys[layer_idx]
            if "jsae" in self.model.saes[key].config.sae_variant:
                if self.model.loss_coefficients.sparsity[layer_idx] == 0 and not is_eval: # compute jacobian loss only on eval if sparsity is 0
                    continue
                topk_indices_mlpin = output.indices[f'{layer_idx}_mlpin']
                topk_indices_mlpout = output.indices[f'{layer_idx}_mlpout']

                mlp_act_grads = output.activations[f"{layer_idx}_mlpactgrads"]

                jacobian_loss = jacobian_mlp(
                    sae_mlpin = self.model.saes[f'{layer_idx}_mlpin'],
                    sae_mlpout = self.model.saes[f'{layer_idx}_mlpout'],
                    mlp = self.model.gpt.transformer.h[layer_idx].mlp,
                    topk_indices_mlpin = topk_indices_mlpin,
                    topk_indices_mlpout = topk_indices_mlpout,
                    mlp_act_grads = mlp_act_grads,
                )

                # Each SAE has it's own loss term, and are trained "independently"
                # so we will put the jacobian loss into the aux loss term
                # for the sae_mlpout for each pair of SAEs
                jacobian_losses[layer_idx] = jacobian_loss
            else:
                warnings.warn(f"JSaeTrainer: Skipping non-JSAE SAE for layer {layer_idx}")

        # Store computed loss components in sparsify output to be read out by gather_metrics
        output.sparsity_losses = jacobian_losses.detach()

        pair_losses = einops.rearrange(recon_losses, "(layer pair) -> layer pair", pair=2).sum(dim=-1)
        losses = pair_losses + j_coeffs * jacobian_losses # (layer)
        return losses


    def gather_metrics(self, loss: torch.Tensor, output: SparsifiedGPTOutput) -> dict[str, torch.Tensor]:
        """
        Gather metrics from loss and model output.
        """
        metrics =  super().gather_metrics(loss, output)
        metrics["∇_l1"] = output.sparsity_losses
        metrics["recon_l2"] = output.recon_losses

        return metrics
