"""
python -m utils.jsae_block --path_pattern="checkpoints/jblock.shk*"
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
from training.sae.jsae_block import JSaeBlockTrainer
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

import glob


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
    parser.add_argument("--config", type=str, help="Training config (default: jsae.shakespeare_64x4)", default="jsae.shakespeare_64x4")
    parser.add_argument("--load_from", type=str, help="GPT model weights to load (default: shakespeare_64x4_dyt)", default="shakespeare_64x4_dyt")
    parser.add_argument("--path_pattern", type=str, help="Path pattern to read models from", required=True)
    return parser.parse_args()


class JSaeBlockEvaler(JSaeBlockTrainer):
    """
    Train SAE weights for all layers concurrently.
    """

    def __init__(self, config: SAETrainingConfig, load_from: str | Path):
        """
        Load and freeze GPT weights before training SAE weights.
        """
        # Create model
        config.sae_config.gpt_config.normalization = NormalizationStrategy.DYNAMIC_TANH
        config.sae_config.compile = False
        model = JBlockSparsifiedGPT(config.sae_config, 
                                  config.loss_coefficients, 
                                  config.trainable_layers)
        #print(model.gpt)
        # Load GPT weights
        #model.load_gpt_weights(load_from)
        #print(f"loading from: {load_from}")
        load_model(model.gpt, os.path.join(load_from, "model.safetensors"), device=config.device.type)

        for idx, sae_key in enumerate(model.saes.keys()):
            #print(f"loading sae {sae_key} from {os.path.join(load_from, f'sae.{idx}.safetensors')}")
            load_model(model.saes[sae_key], os.path.join(load_from, f"sae.{idx}.safetensors"), device=config.device.type)

        # Freeze GPT parameters
        for param in model.parameters():
            param.requires_grad = False
        
        for block in model.gpt.transformer.h:
            assert isinstance(block.ln_2, DynamicTanh), "Only DynamicTanh is supported for JSAE Block"

        SAETrainer.__init__(self, model, config)

        if self.ddp:
            # HACK: We're doing something that causes DDP to crash unless DDP optimization is disabled.
            torch._dynamo.config.optimize_ddp = False  # type: ignore
            
        if self.ddp:
            torch.distributed.barrier()
    
    def train(self):
        """
        Reload model after done training and run eval one more time.
        """
    
        # Wrap the model if using DDP
        if self.ddp:
            self.model = DistributedDataParallel(self.model, device_ids=[self.ddp_local_rank])  # type: ignore

        # Gather final metrics. We don't bother compiling because we're just running eval once.
        print(f"OUTDIR {self.config.out_dir}")
        print(f"NAME {self.config.name}")
        final_metrics = self.val_step(0, should_log=True)  # step 0 so checkpoint isn't saved.
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
        pass
        

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config_name = args.config
    config = options[config_name]
        
    get_paths = glob.glob(args.path_pattern)

    for path in get_paths:
        # Initialize trainer
        path = os.path.normpath(path)
        parts = path.split(os.sep)
        config.name = parts[-1]
        print(f"Evaluating {path}")
        trainer = JSaeBlockEvaler(config, load_from=path)
        trainer.train()
# %%
