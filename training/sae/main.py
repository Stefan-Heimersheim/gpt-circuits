import sys
from pathlib import Path
import torch
import tyro
from dataclasses import dataclass, field
from typing import Optional, Tuple

from config import TrainingConfig
from config.sae.models import SAEVariant
from config.sae.training import SAETrainingConfig, options
from training.sae.concurrent import ConcurrentTrainer
from training.sae.staircase_concurrent import StaircaseConcurrentTrainer
from training.sae.jsae_concurrent import JSaeTrainer


def get_trainer_class(sae_variant: SAEVariant) -> type:
    return {
        SAEVariant.JSAE_BLOCK: JSaeTrainer,
        SAEVariant.JSAE: JSaeTrainer,
        SAEVariant.STAIRCASE_BLOCK: StaircaseConcurrentTrainer,
        SAEVariant.TOPK_STAIRCASE: StaircaseConcurrentTrainer,
        SAEVariant.TOPK_STAIRCASE_DETACH: StaircaseConcurrentTrainer,
        SAEVariant.JUMP_RELU_STAIRCASE: StaircaseConcurrentTrainer,
    }.get(sae_variant, ConcurrentTrainer)


@dataclass
class Args:
    config: str
    load_from: str = "shakespeare_64x4"
    checkpoint_dir: Optional[str] = None

    # TrainingConfig overrides
    name: Optional[str] = None
    device: Optional[str] = None
    compile: Optional[bool] = None
    data_dir: Optional[str] = None
    should_randomize: Optional[bool] = None
    log_interval: Optional[int] = None
    eval_interval: Optional[int] = None
    eval_steps: Optional[int] = None
    batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    warmup_steps: Optional[int] = None
    max_steps: Optional[int] = None
    decay_lr: Optional[bool] = None
    min_lr: Optional[float] = None
    grad_clip: Optional[float] = None
    trainable_layers: Optional[str] = None

    # SAE overrides
    k: Optional[int] = None
    n_features: Optional[str] = None
    sae_keys: Optional[str] = None

    # Loss coefficients
    sparsity: Optional[str] = None
    regularization: Optional[float] = None
    downstream: Optional[float] = None
    bandwidth: Optional[float] = None


def apply_overrides(args: Args, config: SAETrainingConfig):
    simple_fields = {
        "data_dir": config,
        "eval_interval": config,
        "eval_steps": config,
        "batch_size": config,
        "gradient_accumulation_steps": config,
        "learning_rate": config,
        "warmup_steps": config,
        "max_steps": config,
        "decay_lr": config,
        "min_lr": config,
        "log_interval": config,
        "weight_decay": config,
        "grad_clip": config,
        "should_randomize": config,
        "compile": config,
        "downstream": config.loss_coefficients,
        "bandwidth": config.loss_coefficients,
    }

    for field, target in simple_fields.items():
        val = getattr(args, field)
        if val is not None:
            print(f"  Overriding {field} -> {val}")
            setattr(target, field, val)

    if args.device:
        config.device = torch.device(args.device)

    if args.trainable_layers:
        config.trainable_layers = tuple(map(int, args.trainable_layers.split(",")))

    if args.n_features:
        config.sae_config.n_features = tuple(map(int, args.n_features.split(",")))

    if args.sae_keys:
        config.sae_config.sae_keys = tuple(s.strip() for s in args.sae_keys.split(","))

    if args.regularization is not None:
        config.loss_coefficients.regularization = torch.tensor(args.regularization)

    if args.name:
        config.name = args.name

    if args.k is not None and hasattr(config.sae_config, "top_k") and config.sae_config.top_k:
        config.sae_config.top_k = (args.k,) * len(config.sae_config.top_k)
        if not args.name:
            config.name += f".k.{args.k}"

    if args.sparsity:
        vals = list(map(float, args.sparsity.split(",")))
        variant = config.sae_config.sae_variant
        n = len(config.sae_config.top_k) // 2 if variant in {SAEVariant.JSAE, SAEVariant.JSAE_BLOCK} else len(config.sae_config.n_features)
        if len(vals) == 1:
            sparsity = (vals[0],) * n
        elif len(vals) == n:
            sparsity = tuple(vals)
        else:
            raise ValueError(f"--sparsity must have 1 or {n} values, got {len(vals)}")
        config.loss_coefficients.sparsity = sparsity
        if not args.name:
            config.name += f"-sparse-{'_'.join(f'{v:.1e}' for v in vals)}"


if __name__ == "__main__":
    args = tyro.cli(Args)
    config = options[args.config]
    trainer_class = get_trainer_class(config.sae_config.sae_variant)
    print("Applying command-line overrides...")
    apply_overrides(args, config)

    # Use checkpoint_dir override if provided, otherwise use default
    checkpoint_base = args.checkpoint_dir if args.checkpoint_dir else TrainingConfig.checkpoints_dir
    load_path = Path(checkpoint_base) / Path(args.load_from)
    trainer = trainer_class(config, load_from=load_path)

    if isinstance(trainer, JSaeTrainer):
        print(f"{trainer.model.saes.keys()=}")
        print(f"{trainer.model.layer_idxs=}")
        print(f"Using sparsity coefficients: {trainer.model.loss_coefficients.sparsity}")

    print(f"Starting training for {config.name} with {trainer_class.__name__}")
    trainer.train()
