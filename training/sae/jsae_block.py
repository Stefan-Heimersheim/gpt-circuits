"""
python -m training.sae.jsae_block [--config=jsae.shakespeare_64x4] [--load_from=shakespeare_64x4_dyt] [--sparsity=0.02|0.1,0.2,0.3,0.4]
"""
# %%
from pathlib import Path

import einops
import torch
from torch.nn.parallel import DistributedDataParallel

from config.sae.training import SAETrainingConfig
from models.jsaeblockparsified import JBlockSparsifiedGPT
from training.sae.jsae_concurrent import JSaeTrainer
from training.sae import SAETrainer
from safetensors.torch import load_model
import warnings
from typing import Optional, List, Union
from models.gpt import DynamicTanh
from config.gpt.models import NormalizationStrategy
from models.sparsified import SparsifiedGPTOutput
from utils.jsae import jacobian_mlp_block_fast_noeindex

# Change current working directory to parent
# while not os.getcwd().endswith("gpt-circuits"):
#     os.chdir("..")
# print(os.getcwd())


class JSaeBlockTrainer(JSaeTrainer, SAETrainer):
    """
    Train SAE weights for all layers concurrently.
    """
            
    # TODO: This is a very expensive operation, we should try to speed it up
    def output_to_loss(self, output: SparsifiedGPTOutput, is_eval: bool= False) -> torch.Tensor:
        """
        Return an array of losses, one for each pair of layers:
        loss[i] = loss_recon[f"{i}_residmid"] + loss_recon[f"{i}_residpost"] 
            + l1_jacobian(f"{i}_residpost", f"{i}_residmid")
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
                topk_indices_residmid = output.indices[f'{layer_idx}_residmid']
                topk_indices_residpost = output.indices[f'{layer_idx}_residpost']

                mlp_act_grads = output.activations[f"{layer_idx}_mlpactgrads"]

                if self.config.sae_config.gpt_config.norm_strategy == NormalizationStrategy.DYNAMIC_TANH:

                    jacobian_loss = jacobian_mlp_block_fast_noeindex(
                        self.model.saes[f'{layer_idx}_residmid'],
                        self.model.saes[f'{layer_idx}_residpost'],
                        self.model.gpt.transformer.h[layer_idx].mlp,
                        topk_indices_residmid,
                        topk_indices_residpost,
                        mlp_act_grads,
                        norm_act_grads,
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
    
    def train(self):
        """
        Reload model after done training and run eval one more time.
        """
        # Train weights.
        super().train()

        # Wait for all processes to complete training.
        if self.ddp:
            torch.distributed.barrier()

        # Reload all checkpoint weights, which may include those that weren't trained.
        # NOTE: We're using `model_type` to account for use of subclasses.
        # self.model = self.model_type.load(
        #     self.config.out_dir,
        #     loss_coefficients=self.config.loss_coefficients,
        #     trainable_layers=None,  # Load all layers
        #     device=self.config.device,
        # ).to(self.config.device)
        
        #self.model.gpt = self.load_gpt_weights(self.config.out_dir)
        load_model(self.model.gpt, self.config.out_dir / "model.safetensors", device=self.config.device.type)

        # Wrap the model if using DDP
        if self.ddp:
            self.model = DistributedDataParallel(self.model, device_ids=[self.ddp_local_rank])  # type: ignore

        # Gather final metrics. We don't bother compiling because we're just running eval once.
        final_metrics = self.val_step(0, should_log=False)  # step 0 so checkpoint isn't saved.
        self.checkpoint_l0s = final_metrics["l0s"]
        self.checkpoint_ce_loss = final_metrics["ce_loss"]
        self.checkpoint_ce_loss_increases = final_metrics["ce_loss_increases"]
        self.checkpoint_compound_ce_loss_increase = final_metrics["compound_ce_loss_increase"]

        # Summarize results
        if self.is_main_process:
            print(f"Final L0s: {self.pretty_print(self.checkpoint_l0s)}")
            print(f"Final CE loss increases: {self.pretty_print(self.checkpoint_ce_loss_increases)}")
            print(f"Final compound CE loss increase: {self.pretty_print(self.checkpoint_compound_ce_loss_increase)}")
        
    def save_checkpoint(self, model: JBlockSparsifiedGPT, is_best: torch.Tensor):
        """
        Save SAE weights for layers that have achieved a better validation loss.
        """
        # As weights are shared, only save if each layer is better than the previous best.
        return super().save_checkpoint(model, is_best, locs = ('residmid', 'residpost'))
