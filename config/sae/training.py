from dataclasses import dataclass, field
from typing import Optional

import torch

from config import TrainingConfig, map_options

from config.sae.models import SAEConfig, sae_options


@dataclass
class LossCoefficients:
    sparsity: Optional[tuple[float, ...]] = None
    regularization: Optional[torch.Tensor] = None  # For regularization experiment
    downstream: Optional[float] = None  # For end-to-end experiment
    bandwidth: Optional[float] = None  # For JumpReLU


@dataclass
class SAETrainingConfig(TrainingConfig):
    sae_config: SAEConfig = field(default_factory=SAEConfig)
    trainable_layers: Optional[tuple] = None  # If none, all layers are trained.
    loss_coefficients: LossCoefficients = field(default_factory=LossCoefficients)


gpt2_defaults = {
    "data_dir": "data/fineweb_edu_10b",
    "eval_interval": 250,
    "eval_steps": 100,
    "batch_size": 1,
    "gradient_accumulation_steps": 32,
    "learning_rate": 5e-4,
    "warmup_steps": 750,
    "max_steps": 5000,
    "decay_lr": True,
    "min_lr": 1e-4,
}

# Shared training parameters
shakespeare_64x4_defaults = {
    "data_dir": "data/shakespeare",
    "eval_interval": 250,
    "eval_steps": 100,
    "batch_size": 128,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-3,
    "warmup_steps": 750,
    "max_steps": 7500,
    "decay_lr": True,
    "min_lr": 1e-4,
}
stories_32x4_defaults = {
    "data_dir": "data/tiny_stories_10m",
    "eval_interval": 250,
    "eval_steps": 100,
    "batch_size": 128,
    "gradient_accumulation_steps": 2,
    "learning_rate": 1e-3,
    "max_steps": 5000,
    "decay_lr": True,
    "min_lr": 1e-4,
}
stories_256x4_defaults = {
    "data_dir": "data/tiny_stories",
    "eval_interval": 250,
    "eval_steps": 100,
    "batch_size": 256,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-3,
    "max_steps": 5000,
    "decay_lr": True,
    "min_lr": 1e-4,
}

# Shared training parameters
tiny_defaults = {
    "data_dir": "data/tiny_stories_10m",
    "eval_interval": 250,
    "eval_steps": 100,
    "batch_size": 128,
    "gradient_accumulation_steps": 2,
    "learning_rate": 1e-3,
    "max_steps": 5000,
    "decay_lr": True,
    "min_lr": 1e-4,
}


# Training configuration options
options: dict[str, SAETrainingConfig] = map_options(
    # No such thing as tiny_64x2?
    # SAETrainingConfig(
    #     name="standard.tiny_64x2",
    #     sae_config=sae_options["standardx8.tiny_64x2"],
    #     **tiny_defaults,
    #     loss_coefficients=LossCoefficients(
    #     sparsity=(0.02, 0.06, 0.2, 0.2, 0.5),  # Targets L0s of ~10
    #     ),
    # ),
    SAETrainingConfig(
        name="topk.tblock.gpt2",
        sae_config=sae_options["topk.tblock.gpt2"],
        **gpt2_defaults,
        loss_coefficients=LossCoefficients(),
    ),
    SAETrainingConfig(
        name="jln.mlpblock.gpt2",
        sae_config=sae_options["jln.mlpblock.gpt2"],
        **gpt2_defaults,
        loss_coefficients=LossCoefficients(
            sparsity=(0.5)*24,
        ),
    ),
    SAETrainingConfig(
        name="jsae.mlp_ln.shk_64x4",
        sae_config=sae_options["jsae.mlp_ln.shk_64x4"],
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(
            sparsity=(0.02, 0.02, 0.02, 0.02),
        ),
    ),
    SAETrainingConfig(
        name="staircase.tblock.gpt2",
        sae_config=sae_options["staircase.tblock.gpt2"],
        **gpt2_defaults,
        loss_coefficients=LossCoefficients(),
    ),
    SAETrainingConfig(
        name="topk.mlplayer.shk_64x4",
        sae_config=sae_options["topk.mlplayer.shakespeare_64x4"],
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(),
    ),
    SAETrainingConfig(
        name="topk.mlpblock.shk_64x4",
        sae_config=sae_options["topk.mlpblock.shakespeare_64x4"],
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(),
    ),
    SAETrainingConfig(
        name="topk.tblock.shk_64x4",
        sae_config=sae_options["topk.tblock.shakespeare_64x4"],
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(),
    ),
    SAETrainingConfig(
        name="staircase-mlpblock.shk_64x4",
        sae_config=sae_options["staircase-pairsx8.shakespeare_64x4"],
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(),
    ),
    SAETrainingConfig(
        name="topk.stair.tblock.shk_64x4",
        sae_config=sae_options["topk-staircase-10-x8-share.shakespeare_64x4"],
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(),
    ),
    
    # SAETrainingConfig(
    #     name="mlp-topk.shakespeare_64x4",
    #     sae_config=sae_options["mlp.topkx8.shakespeare_64x4"],
    #     **shakespeare_64x4_defaults,
    #     loss_coefficients=LossCoefficients(),
    # ),
    SAETrainingConfig(
        name="jsae.mlp_layer.shk_64x4",
        sae_config=sae_options["jsae.topkx8.shakespeare_64x4"],
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(
            sparsity=(0.02, 0.02, 0.02, 0.02),  # How do we tune the sparsity? What is the target?
        ),
    ),
     SAETrainingConfig(
        name="standard.tiny_32x4",
        sae_config=sae_options["standardx16.tiny_32x4"],
        **tiny_defaults,
        loss_coefficients=LossCoefficients(
            sparsity=(0.06, 0.06, 0.1, 0.1, 0.1),  # Targets L0s of ~10
        ),
    ),
    SAETrainingConfig(
        name="multilayer.shakespeare_64x4",
        sae_config=sae_options["staircasex8.shakespeare_64x4"],
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(
            sparsity=(0.02, 0.06, 0.2, 0.2, 0.5),  # Targets L0s of ~10
        ),
    ),
    SAETrainingConfig(
        name="standard.shakespeare_64x4",
        sae_config=sae_options["standardx8.shakespeare_64x4"],
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(
            sparsity=(0.02, 0.06, 0.2, 0.2, 0.5),  # Targets L0s of ~10
        ),
    ),
    SAETrainingConfig(
        name="topk.shakespeare_64x4",
        sae_config=sae_options["topk-10-x8.shakespeare_64x4"],
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients()
    ),
    SAETrainingConfig(
        name="standard_fat.shakespeare_64x4",
        sae_config=sae_options["standardx40.shakespeare_64x4"],
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(
            sparsity=(0.02, 0.06, 0.2, 0.2, 0.5),  # Targets L0s of ~10
        ),
    ),
    SAETrainingConfig(
        name="staircase.shakespeare_64x4",
        sae_config=sae_options["staircasex8.shakespeare_64x4"],
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(
            sparsity=(0.06, 0.24, 0.8, 1.0, 2.5) # Targets L0s of ~10
        ),
    ),
    SAETrainingConfig(
        name="topk-staircase-noshare.shakespeare_64x4",
        sae_config=sae_options["topk-staircase-10-x8-noshare.shakespeare_64x4"],
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(),
    ),

    SAETrainingConfig(
        name="topk-staircase-detach.shakespeare_64x4",
        sae_config=sae_options["topk-staircase-10-x8-detach.shakespeare_64x4"],
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(),
    ),
    SAETrainingConfig(
        name="top5.shakespeare_64x4",
        sae_config=sae_options["top5.shakespeare_64x4"],
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(),
    ),
    SAETrainingConfig(
        name="top20.shakespeare_64x4",
        sae_config=sae_options["top20.shakespeare_64x4"],
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(),
    ),
    SAETrainingConfig(
        name="regularized.shakespeare_64x4",
        sae_config=sae_options["standardx8.shakespeare_64x4"],
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(
            sparsity=(0.020, 0.035, 0.085, 0.07, 0.075),
            regularization=torch.tensor(3.0),
        ),
    ),
    SAETrainingConfig(
        name="end-to-end.shakespeare_64x4",
        sae_config=sae_options["standardx8.shakespeare_64x4"],
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(
            sparsity=(0.50, 0.80, 0.80, 0.15, 0.80),
            downstream=1.0,
        ),
    ),
    SAETrainingConfig(
        name="jumprelu.shakespeare_64x4",
        sae_config=sae_options["jumprelu-x8.shakespeare_64x4"],
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(
            sparsity=(0.01, 0.01, 0.01, 0.03, 0.08),
            bandwidth=0.1,
        ),
    ),
    SAETrainingConfig(
        name="jumprelu-staircase.shakespeare_64x4",
        sae_config=sae_options["jumprelu-staircase-x8.shakespeare_64x4"],
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(
            sparsity=(0.01, 0.01, 0.01, 0.03, 0.08),
            bandwidth=0.1,
        ),
    ),
    SAETrainingConfig(
        name="e2e.jumprelu.shakespeare_64x4",
        sae_config=sae_options["jumprelu-x16.shakespeare_64x4"],
        **(
            shakespeare_64x4_defaults
            | {
                "max_steps": 10000,
                "min_lr": 1e-5,
            }
        ),
        loss_coefficients=LossCoefficients(
            sparsity=(0.01, 0.00001, 0.005, 0.01, 0.005),
            downstream=1.0,
            bandwidth=0.1,
        ),
    ),
    SAETrainingConfig(
        name="jumprelu.stories_256x4",
        sae_config=sae_options["jumprelu-x32.stories_256x4"],
        **stories_256x4_defaults,
        loss_coefficients=LossCoefficients(
            # L0s ≈ (6...20)
            sparsity=(0.05, 0.1, 0.2, 0.4, 2.0),
            bandwidth=0.1,
        ),
    ),
    SAETrainingConfig(
        name="e2e.jumprelu.stories_256x4",
        sae_config=sae_options["jumprelu-x32.stories_256x4"],
        **stories_256x4_defaults
        | {
            "batch_size": 64,
            "gradient_accumulation_steps": 32,
        },
        loss_coefficients=LossCoefficients(
            # L0s ≈ (10...30), CE Loss Increase < 0.1
            sparsity=(0.002, 0.10, 0.16, 0.005, 0.005),
            downstream=1.0,
            bandwidth=0.1,
        ),
    ),
)
