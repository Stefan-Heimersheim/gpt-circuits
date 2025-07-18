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
from config.sae.models import SAEVariant, gen_sae_locations, HookPoint
from config.gpt.models import NormalizationStrategy
import warnings
from typing import Optional


# Change current working directory to parent
# while not os.getcwd().endswith("gpt-circuits"):
#     os.chdir("..")
# print(os.getcwd())

from utils.jsae import jacobian_mlp, jacobian_mlp_block_fast_noeindex, jacobian_mlp_block_ln


class JSaeTrainer(ConcurrentTrainer):
    """
    Train SAE weights for all layers concurrently.
    """
    def __init__(self, config: SAETrainingConfig, load_from: str | Path):
        super().__init__(config, load_from)
        self.locs = gen_sae_locations(config.sae_config.sae_variant)
            
    def save_checkpoint(self, model: JSparsifiedGPT, is_best: torch.Tensor):
        """
        Save SAE weights for layers that have achieved a better validation loss.
        """
        # SAEs are trained in pairs. Each layer has a loss, save the pair is loss is better.
        sae_keys_to_save = []
        for layer_idx in self.model.layer_idxs:
            if is_best[layer_idx]:
                sae_keys_to_save.append(f'{layer_idx}_{self.locs.in_loc}') # in
                sae_keys_to_save.append(f'{layer_idx}_{self.locs.out_loc}') # out
                
        model.save(self.config.out_dir, sae_keys_to_save)

    # TODO: This is a very expensive operation, we should try to speed it up
    def raw_jacobian_losses(self, output: SparsifiedGPTOutput, is_eval: bool= False) -> torch.Tensor:
        device = output.sae_losses.device
        jacobian_losses = torch.zeros(len(self.model.layer_idxs), device=device)
         
        for layer_idx in self.model.layer_idxs:
            key = self.model.sae_keys[layer_idx]
            if not SAEVariant(self.model.saes[key].config.sae_variant).is_jsae():
                warnings.warn(f"JSaeTrainer: Skipping non-JSAE SAE for layer {layer_idx}")
                continue
        
            # if sparsity is 0, only compute jacobian during evaluation
            if self.model.loss_coefficients.sparsity[layer_idx] == 0 and not is_eval:
                continue
            
            block = self.model.gpt.transformer.h[layer_idx]
            mlp = block.mlp
            mlp_act_grads = output.activations[f"{layer_idx}_{HookPoint.MLP_ACT_GRAD.value}"]
            idx_in = output.indices[f'{layer_idx}_{self.locs.in_loc}']
            idx_out = output.indices[f'{layer_idx}_{self.locs.out_loc}']
            sae_in = self.model.saes[f'{layer_idx}_{self.locs.in_loc}']
            sae_out = self.model.saes[f'{layer_idx}_{self.locs.out_loc}']
            
            if self.config.sae_config.sae_variant == SAEVariant.JSAE_LAYER:
                
                j_loss = jacobian_mlp(sae_in, sae_out, mlp, idx_in, idx_out, mlp_act_grads)
                
            elif self.config.sae_config.sae_variant == SAEVariant.JSAE_BLOCK:
                
                if self.config.sae_config.gpt_config.norm_strategy == NormalizationStrategy.DYNAMIC_TANH:
                    dyt_grads = output.activations[f"{layer_idx}_{HookPoint.DYT_ACT_GRAD.value}"]
                    j_loss = jacobian_mlp_block_fast_noeindex(sae_in, sae_out, mlp, idx_in, idx_out, mlp_act_grads, dyt_grads)
                    
                elif self.config.sae_config.gpt_config.norm_strategy == NormalizationStrategy.LAYER_NORM:
                    resid_mid = output.activations[f"{layer_idx}_{HookPoint.RESID_MID.value}"]
                    _, ln_pre_x, ln_scale = block.ln_2(resid_mid, return_std=True)
                    gamma = block.ln_2.weight
                    j_loss = jacobian_mlp_block_ln(sae_in, sae_out, mlp, idx_in, idx_out, mlp_act_grads, gamma, ln_pre_x, ln_scale)
                
                else:
                    raise ValueError(f"Jacobian not defined for normalization strategy: {self.config.sae_config.gpt_config.norm_strategy}")
                
            else:
                raise ValueError(f"Jacobian not defined for SAE variant: {self.config.sae_config.sae_variant}")
                

            # Each SAE has it's own loss term, and are trained "independently"
            # so we will put the jacobian loss into the aux loss term
            # for the sae_mlpout for each pair of SAEs
            assert j_loss.ndim == 0, f"Expected scalar tensor for j_loss, got shape {j_loss.shape}"
            jacobian_losses[layer_idx] = j_loss

        return jacobian_losses

    # TODO: This is a very expensive operation, we should try to speed it up
    def output_to_loss(self, output: SparsifiedGPTOutput, is_eval: bool= False) -> torch.Tensor:
        """
        Return an array of losses, one for each pair of layers:
        
        loss[i] = loss_recon[f"{i}_{locin}"] 
                + loss_recon[f"{i}_{locout}"] 
                + j_coeff[i] * loss_jacobian(f"{i}_{locout}", f"{i}_{locin}")
            
        For JSAE,       (locin, locout) = ("mlpin", "mlpout")
        For JSAE_BLOCK, (locin, locout) = ("residmid", "residpost")
        """
        device = output.sae_losses.device
        
        recon_losses = output.sae_losses
        jacobian_losses = self.raw_jacobian_losses(output, is_eval) # returns jacobian losses without sparsity coefficient

        # Store computed loss components in sparsify output to be read out by gather_metrics
        output.sparsity_losses = jacobian_losses.detach()
        output.aux_losses = jacobian_losses.detach()
        
        # Recon losses are per SAE, so we need to pair adjacent losses together
        # pair_losses[i] = recon_losses[2*i] + recon_losses[2*i+1]
        #                = loss_recon[f"{i}_{locin}"]  + loss_recon[f"{i}_{locout}"]
        pair_losses = einops.rearrange(recon_losses, "(layer pair) -> layer pair", pair=2).sum(dim=-1)
        assert pair_losses.shape == jacobian_losses.shape,(
            f"jsae_concurrent.output_to_loss {pair_losses.shape=}, {jacobian_losses.shape=}")
        
        j_coeffs = torch.tensor(self.model.loss_coefficients.sparsity, device=device)
        assert j_coeffs.shape == jacobian_losses.shape,( 
            f"jsae_concurrent: jacobian_losses.shape: {jacobian_losses.shape}, j_coeffs.shape: {j_coeffs.shape}")
        
        scaled_jacobian_losses = j_coeffs * jacobian_losses # (layer)
        self.scaled_jacobian_losses = scaled_jacobian_losses.detach()
        
        losses = pair_losses + scaled_jacobian_losses # (layer)
        return losses


    def gather_metrics(self, loss: torch.Tensor, output: SparsifiedGPTOutput) -> dict[str, torch.Tensor]:
        """
        Gather metrics from loss and model output.
        """
        metrics =  super().gather_metrics(loss, output)
        metrics["∇_l1_raw"] = output.sparsity_losses
        metrics["∇_l1"] = self.scaled_jacobian_losses
        metrics["recon_l2"] = output.recon_losses

        return metrics
