from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TypeVar

import torch


@dataclass
class Config:
    name: str = field(default="")
    device: torch.device = field(default_factory=lambda: get_default_device())
    compile: bool = field(default_factory=lambda: get_default_device().type == "cuda")


@dataclass
class TrainingConfig(Config):
    # Data parameters
    data_dir: str = ""
    should_randomize: bool = True

    # Evaluation parameters
    log_interval: int = 10
    eval_interval: int = 0
    eval_steps: int = 0

    # Batch parameters
    batch_size: int = 0
    gradient_accumulation_steps: int = 0

    # Training parameters
    learning_rate: float = 0
    warmup_steps: int = 0
    max_steps: int = 0
    decay_lr: bool = False
    min_lr: float = 0  # Minimum learning rate
    weight_decay: float = 0.1
    grad_clip: Optional[float] = 1.0  # Maximum gradient norm

    @property
    def out_dir(self) -> Path:
        """
        Checkpoint path
        """
        return Path(self.checkpoints_dir) / Path(self.name)

    checkpoints_dir = "checkpoints"


ConfigType = TypeVar("ConfigType", bound=Config)


def get_default_device() -> torch.device:
    """
    Defaults to CPU if no GPU is available.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def map_options(*options: ConfigType) -> dict[str, ConfigType]:
    """
    Map configurations to a dictionary using name as key.
    """
    return {option.name: option for option in options}
