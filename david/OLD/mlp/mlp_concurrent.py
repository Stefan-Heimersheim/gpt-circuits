"""
python -m training.sae.mlp_concurrent --config=mlp.standard.shakespeare_64x4 --load_from=shakespeare_64x4
"""
import argparse
from pathlib import Path

import torch

from config import TrainingConfig
from config.sae.training import SAETrainingConfig, options
from models.sparsified import SparsifiedGPTOutput
from david.OLD.mlp.sparsified_mlp import SparsifiedMLPGPT
from training.sae import SAETrainer


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Training config")
    parser.add_argument("--load_from", type=str, help="GPT model weights to load")
    parser.add_argument("--name", type=str, help="Model name for checkpoints")
    return parser.parse_args()


class MLPConcurrentTrainer(SAETrainer):
    """
    Train SAE weights for all layers concurrently using Sparsified MLP GPT model.
    """
    def __init__(self, config: SAETrainingConfig, load_from: str | Path):
        """
        Load and freeze GPT weights before training SAE weights.
        """
        # Create model
        model = SparsifiedMLPGPT(config.sae_config, config.loss_coefficients, config.trainable_layers)

        # Load GPT weights
        model.load_gpt_weights(load_from)

        # Freeze GPT parameters
        for param in model.gpt.parameters():
            param.requires_grad = False

        super().__init__(model, config)

        if self.ddp:
            # HACK: We're doing something that causes DDP to crash unless DDP optimization is disabled.
            torch._dynamo.config.optimize_ddp = False  # type: ignore

    def output_to_loss(self, output: SparsifiedGPTOutput) -> torch.Tensor:
        """
        Return an array of losses instead of a single combined loss.
        """
        return output.sae_losses

    def backward(self, loss):
        """
        Because SAE layers are independent, we can add layer losses and run a single backward pass instead of having to
        run a separate backward pass using each layer's loss. The results are equivalent.
        """
        loss.sum().backward()

    def save_checkpoint(self, model: SparsifiedMLPGPT, is_best: torch.Tensor):
        """
        Save SAE weights for layers that have achieved a better validation loss.
        """
        # `is_best` contains a value for each layer indicating whether we have the best loss for that layer.
        layers_to_save = [layer_name for should_save, layer_name in zip(is_best, model.saes.keys()) if should_save]
        model.save(self.config.out_dir, layers_to_save)


def main():
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config_name = args.config
    config = options[config_name]
    assert "staircase" not in config.sae_config.sae_variant, "Staircase SAE variant is not supported for concurrent training."
    # Update outdir
    if args.name:
        config.name = args.name

    # Initialize trainer
    trainer = MLPConcurrentTrainer(config, load_from=TrainingConfig.checkpoints_dir / args.load_from)
    trainer.train()

if __name__ == "__main__":
    main()