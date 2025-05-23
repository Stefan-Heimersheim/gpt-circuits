"""
python -m training.sae.jsae_block [--config=jsae.shakespeare_64x4] [--load_from=shakespeare_64x4_dyt] [--sparsity=0.02|0.1,0.2,0.3,0.4]
"""
# %%
import argparse
import os
import sys
from pathlib import Path

import einops
import torch

from config import TrainingConfig
from config.sae.training import SAETrainingConfig, options
from models.gpt import MLP, DynamicTanh
from models.jsaeblockparsified import JBlockSparsifiedGPT
from training.sae.jsae_concurrent import JSaeTrainer
from models.sae import SparseAutoencoder
from models.sparsified import SparsifiedGPTOutput
from training.sae import SAETrainer
from training.sae.concurrent import ConcurrentTrainer
from config.gpt.models import GPTConfig
from models.gpt import GPT
from config.gpt.models import NormalizationStrategy
from safetensors.torch import load_model
import dataclasses
import json
from config.sae.models import SAEConfig
from training import Trainer
import warnings
from typing import Optional, List, Union
# Change current working directory to parent
# while not os.getcwd().endswith("gpt-circuits"):
#     os.chdir("..")
# print(os.getcwd())

from utils.jsae import jacobian_mlp_block_fast_noeindex

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Training config", default="jsae.shakespeare_64x4")
    parser.add_argument("--load_from", type=str, help="GPT model weights to load", default="shakespeare_64x4_dyt")
    parser.add_argument("--name", type=str, help="Model name for checkpoints", default="jblock.shk_64x4")
    parser.add_argument("--sparsity", type=str, help="Jacobian sparsity loss coefficient(s). Either a single float or comma-separated floats for each layer.", default="0.02")
    parser.add_argument("--max_steps", type=int, help="Maximum number of steps to train", default=20000)
    parser.add_argument("--k", type=int, help="Top-k for top-k SAEs", default=10)
    return parser.parse_args()


class JSaeBlockTrainer(JSaeTrainer, Trainer):
    """
    Train SAE weights for all layers concurrently.
    """

    def __init__(self, config: SAETrainingConfig, load_from: str | Path):
        """
        Load and freeze GPT weights before training SAE weights.
        """
        # Create model
        print(config.sae_config.gpt_config.norm_strategy)
        model = JBlockSparsifiedGPT(config.sae_config, 
                                  config.loss_coefficients, 
                                  config.trainable_layers)

        print(model.gpt)
        # Load GPT weights
        #model.load_gpt_weights(load_from)
        print(f"loading from: {load_from}")
        load_model(model.gpt, load_from / "model.safetensors", device=config.device.type)

        # Freeze GPT parameters
        for param in model.gpt.parameters():
            param.requires_grad = False
        
        for block in model.gpt.transformer.h:
            if not isinstance(block.ln_2, DynamicTanh):
                warnings.warn(f"JSaeBlockTrainer: Expecting DynamicTanh after ln_2, but got {type(block.ln_2)}")

        SAETrainer.__init__(self, model, config)

        if self.ddp:
            # HACK: We're doing something that causes DDP to crash unless DDP optimization is disabled.
            torch._dynamo.config.optimize_ddp = False  # type: ignore
    
    def train(self):
        """
        Reload model after done training and run eval one more time.
        """
        # Train weights.
        Trainer.train(self)

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

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config_name = args.config
    config = options[config_name]

    if args.max_steps:
        config.max_steps = args.max_steps

    if args.sparsity is not None:
        try:
            sparsity_values_str = args.sparsity.split(',')
            sparsity_values = [float(s.strip()) for s in sparsity_values_str]
        except ValueError:
            print(f"Error: Invalid format for --sparsity. Expected a single float or comma-separated floats. Got: {args.sparsity}")
            sys.exit(1)

        num_trainable_layers = len(config.sae_config.top_k)//2 # Use initial config to know layer count

        if len(sparsity_values) == 1:
            # Single value provided, apply to all layers
            sparsity_val = sparsity_values[0]
            config.loss_coefficients.sparsity = (sparsity_val,) * num_trainable_layers
            print(f"Setting sparsity to {sparsity_val} for all {num_trainable_layers} layers")
            config.name = f"{config.name}-sparse-{sparsity_val:.1e}"
        elif len(sparsity_values) == num_trainable_layers:
            # Correct number of values provided for each layer
            config.loss_coefficients.sparsity = tuple(sparsity_values)
            print(f"Setting per-layer sparsity: {config.loss_coefficients.sparsity}")
            # Modify name to include all sparsity values
            sparsity_str = "_".join([f"{s:.1e}" for s in sparsity_values])
            config.name = f"{config.name}-sparse-{sparsity_str}"
        else:
            # Incorrect number of values
            raise ValueError(f"Error: --sparsity must provide either 1 value (for all layers) or {num_trainable_layers} values (one per layer). Got {len(sparsity_values)} values.")

    if args.k:
        config.k = args.k
        config.name = f"{config.name}-k-{args.k}"
        
    if args.name:
        config.name = args.name

    # Initialize trainer
    trainer = JSaeBlockTrainer(config, load_from=TrainingConfig.checkpoints_dir / args.load_from)
    print(f'{trainer.model.saes.keys()=}')
    print(f'{trainer.model.layer_idxs=}')
    print(f'Using sparsity coefficients: {trainer.model.loss_coefficients.sparsity}')
    trainer.train()

# %%
