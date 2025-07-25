from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional

from config import Config, map_options
from config.gpt.models import GPTConfig, gpt_options
from collections import namedtuple
import warnings

class HookPoint(str, Enum):
    RESID_PRE = "act"
    RESID_MID = "residmid"
    RESID_POST = "residpost"
    MLP_IN = "mlpin"
    MLP_OUT = "mlpout"
    ACT = "act" #alias for RESID_PRE
    ATTN_OUT = "attnout"
    ATTN_IN = "attnin"
    MLP_ACT_GRAD = "mlpactgrads"
    DYT_ACT_GRAD = "normactgrads"
    
    @classmethod
    def all(cls) -> list[str]:
        return [loc.value for loc in cls]

class SAEVariant(str, Enum):
    STANDARD = "standard"
    STANDARD_V2 = "standard_v2"
    GATED = "gated"
    GATED_V2 = "gated_v2"
    JUMP_RELU = "jumprelu"
    JUMP_RELU_STAIRCASE = "jumprelu_staircase"
    TOPK = "topk"
    TOPK_STAIRCASE = "topk_staircase"
    TOPK_STAIRCASE_DETACH = "topk_staircase_detach"
    JSAE = "jsae_layer" # deprecated. Don't use this.
    JSAE_LAYER = "jsae_layer"
    JSAE_BLOCK = "jsae_block"
    STAIRCASE_BLOCK = "staircase_block"
    
    def is_jsae(self) -> bool:
        """Check if this variant is a JSAE type."""
        return self in {SAEVariant.JSAE, SAEVariant.JSAE_LAYER, SAEVariant.JSAE_BLOCK}
    


SAELocations = namedtuple('SAELocations', ['in_loc', 'out_loc'])

location_map: dict[SAEVariant, SAELocations] = {
    SAEVariant.JSAE_LAYER: SAELocations(HookPoint.MLP_IN.value, HookPoint.MLP_OUT.value),
    SAEVariant.JSAE_BLOCK: SAELocations(HookPoint.RESID_MID.value, HookPoint.RESID_POST.value),
}

def gen_sae_locations(sae_variant: SAEVariant) -> SAELocations:
    """
    Generate the location of the SAE weights to train.
    """
    if sae_variant not in location_map:
        raise ValueError(f"gen_sae_locations: Unsupported SAE variant: {sae_variant}")
    return location_map[sae_variant]
    
@dataclass
class SAEConfig(Config):
    gpt_config: GPTConfig = field(default_factory=GPTConfig)
    n_features: tuple = ()  # Number of features in each layer
    sae_variant: SAEVariant = SAEVariant.STANDARD
    top_k: Optional[tuple[int, ...]] = None  # Required for topk variant
    sae_keys: Optional[tuple[str, ...]] = None
    deprecated: Optional[str] = None

    def __post_init__(self):
        if self.deprecated is not None:
            warnings.warn(self.deprecated, DeprecationWarning, stacklevel=2)

    @property
    def block_size(self) -> int:
        return self.gpt_config.block_size

    @staticmethod
    def dict_factory(fields: list) -> dict:
        """
        Only export n_features and sae_variant.
        """
        whitelisted_fields = ("n_features", "sae_variant", "top_k", "sae_keys")
        return {k: v for (k, v) in fields if k in whitelisted_fields and v is not None}

def gen_sae_keys(n_features: int, loc : Literal["mlplayer", "mlpblock", "standard"] = 'standard') -> tuple[str, ...]:
    match loc:
        case "mlplayer":
            assert n_features % 2 == 0, "n_features must be even for mlpplayer: " + str(n_features)
            return tuple(f"{x}_{y}" for x in range(n_features//2) for y in [HookPoint.MLP_IN.value, HookPoint.MLP_OUT.value])
        case "mlpblock":
            assert n_features % 2 == 0, "n_features must be even for mlpblock: " + str(n_features)
            return tuple(f"{x}_{y}" for x in range(n_features//2) for y in [HookPoint.RESID_MID.value, HookPoint.RESID_POST.value])
        case _:
            # assume we're using the activations between the transformer blocks
            # for a 4 layer transformer, this will be:
            # 0_act = 0_residpre (i.e. between the embedding layer and the first transformer block)
            # 1_act = 1_residpre (i.e. between the first and second transformer blocks)
            # 2_act = 2_residpre (i.e. between the second and third transformer blocks)
            # 3_act = 3_residpre (i.e. between the third and fourth transformer blocks)
            # 4_act = 3_residpost (i.e. between the fourth/last transformer block and the final layer norm)
            return tuple(f"{x}_{HookPoint.ACT.value}" for x in range(n_features))

# SAE configuration options
sae_options: dict[str, SAEConfig] = map_options(
    SAEConfig("topk.tblock.gpt2",
        gpt_config = gpt_options['gpt2'],
        n_features=tuple(768 * n for n in (32,)*13),
        sae_variant=SAEVariant.TOPK,
        top_k = (32,) * 13,
        sae_keys=gen_sae_keys(n_features=13, loc="standard"),
    ),
    SAEConfig(
        name="jln.mlpblock.gpt2",
        gpt_config = gpt_options['gpt2'],
        n_features=tuple(768 * n for n in (32,)*24), # 2 per layer, 12 layers
        sae_variant=SAEVariant.JSAE_BLOCK,
        top_k = (32,) * 24,
        sae_keys=gen_sae_keys(n_features=24, loc="mlpblock"),
    ),
    SAEConfig(
        name="jsae.mlp_ln.shk_64x4",
        gpt_config = gpt_options['ascii_64x4'],
        n_features=tuple(64 * n for n in (8, 8, 8, 8, 8, 8, 8, 8)),
        sae_variant=SAEVariant.JSAE_BLOCK,
        top_k = (10, 10, 10, 10, 10, 10, 10, 10),
        sae_keys=gen_sae_keys(n_features=8, loc="mlpblock"),
    ),
    SAEConfig(
        name="staircase.tblock.gpt2",
        gpt_config=gpt_options['gpt2'],
        n_features=tuple(768 * n for n in range(32, 32*13+1, 32)),
        sae_variant=SAEVariant.TOPK_STAIRCASE,
        top_k = (32,) * 13,
        sae_keys=gen_sae_keys(n_features=13, loc="standard"),
    ),
    SAEConfig(
        name="mlp.standardx8.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 8, 8, 8, 8, 8, 8, 8)),
        sae_variant=SAEVariant.STANDARD,
        sae_keys=gen_sae_keys(n_features=8, loc="mlplayer"),
    ),
    SAEConfig(
        name="topk.mlplayer.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8,8,8,8,8,8,8,8)),
        sae_variant=SAEVariant.TOPK,
        top_k=(10, 10, 10, 10, 10, 10, 10, 10),
        sae_keys=gen_sae_keys(n_features=8, loc="mlplayer"),
    ),
    SAEConfig(
        name="topk.mlpblock.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8,8,8,8,8,8,8,8)),
        sae_variant=SAEVariant.TOPK,
        top_k=(10, 10, 10, 10, 10, 10, 10, 10),
        sae_keys=gen_sae_keys(n_features=8, loc="mlpblock"),
    ),
    SAEConfig(
        name="topk.tblock.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8,8,8,8,8)),
        sae_variant=SAEVariant.TOPK,
        top_k=(10, 10, 10, 10, 10),
        sae_keys=gen_sae_keys(n_features=8, loc="standard"),
    ),
    SAEConfig(
        name="staircase-pairsx8.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 16, 8, 16, 8, 16, 8, 16)),
        sae_variant=SAEVariant.STAIRCASE_BLOCK,
        top_k=(10, 10, 10, 10, 10, 10, 10, 10),
        sae_keys=gen_sae_keys(n_features=8, loc="mlpblock"),
    ),
    SAEConfig(
        name="staircase.mlpblock.gpt2",
        gpt_config=gpt_options['gpt2'],
        n_features=tuple(768 * n for n in 13*(32,64)),
        sae_variant=SAEVariant.STAIRCASE_BLOCK,
        top_k = (32,32) * 13,
        sae_keys=gen_sae_keys(n_features=26, loc="mlpblock"),
    ),
    SAEConfig(
        name="mlp_layer.topkx8.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 8, 8, 8, 8, 8, 8, 8)),
        sae_variant=SAEVariant.TOPK,
        top_k=(10, 10, 10, 10, 10, 10, 10, 10),
        sae_keys=gen_sae_keys(n_features=8, loc="mlplayer"),
        deprecated="Use 'topk.mlplayer.shakespeare_64x4' instead",
    ),
    SAEConfig(
        name="jsae.topkx8.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 8, 8, 8, 8, 8, 8, 8)),
        sae_variant=SAEVariant.JSAE_LAYER,
        top_k=(10, 10, 10, 10, 10, 10, 10, 10),
        sae_keys=gen_sae_keys(n_features=8, loc="mlplayer"),
    ),
    SAEConfig(
        name="standardx16.tiny_32x4",
        gpt_config=gpt_options["tiktoken_32x4"],
        n_features=tuple(32 * n for n in (16, 16, 16, 16, 16)),
        sae_variant=SAEVariant.STANDARD,
        sae_keys=gen_sae_keys(n_features=5),
    ),
    # No such thing as tiny_64x2?
    # SAEConfig(
    #     name="standardx8.tiny_64x2",
    #     gpt_config=gpt_options["tiktoken_64x2"],
    #     n_features=tuple(64 * n for n in (8, 8, 8, 8, 8)),
    #     sae_variant=SAEVariant.STANDARD,
    # ),
    SAEConfig(
        name="standardx8.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 8, 8, 8, 8)),
        sae_variant=SAEVariant.STANDARD,
        sae_keys=gen_sae_keys(n_features=5),
    ),
    SAEConfig(
        name="topk-10-x8.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 8, 8, 8, 8)),
        sae_variant=SAEVariant.TOPK,
        top_k=(10, 10, 10, 10, 10),
        sae_keys=gen_sae_keys(n_features=5),
    ),
    SAEConfig(
        name="topk-staircase-10-x8-noshare.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 16, 24, 32, 40)),
        sae_variant=SAEVariant.TOPK,
        top_k=(10, 10, 10, 10, 10),
        sae_keys=gen_sae_keys(n_features=5),
    ),
    SAEConfig(
        name="topk-staircase-10-x8-share.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 16, 24, 32, 40)),
        sae_variant=SAEVariant.TOPK_STAIRCASE,
        top_k=(10, 10, 10, 10, 10),
        sae_keys=gen_sae_keys(n_features=5),
    ),
    SAEConfig(
        name="topk-staircase-10-x8-detach.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 16, 24, 32, 40)),
        sae_variant=SAEVariant.TOPK_STAIRCASE_DETACH,
        top_k=(10, 10, 10, 10, 10),
        sae_keys=gen_sae_keys(n_features=5),
    ),
    SAEConfig(
        name="standardx40.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (40, 40, 40, 40, 40)),
        sae_variant=SAEVariant.STANDARD,
        sae_keys=gen_sae_keys(n_features=5),
    ),
    SAEConfig(
        name="staircasex8.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 16, 24, 32, 40)),
        sae_variant=SAEVariant.STANDARD,
        sae_keys=gen_sae_keys(n_features=5),
    ),
    SAEConfig(
        name="topk-x8.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 8, 8, 8, 8)),
        sae_variant=SAEVariant.TOPK,
        top_k=(10,10,10,10,10),
        sae_keys=gen_sae_keys(n_features=5),
    ),
    SAEConfig(
        name="top5.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 8, 8, 8, 8)),
        sae_variant=SAEVariant.TOPK,
        top_k=(5,5,5,5,5),
        sae_keys=gen_sae_keys(n_features=5),
    ),
     SAEConfig(
        name="top20.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 8, 8, 8, 8)),
        sae_variant=SAEVariant.TOPK,
        top_k=(20,20,20,20,20),
        sae_keys=gen_sae_keys(n_features=5),
    ),
    SAEConfig(
        name="topk-x40.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (40, 40, 40, 40, 40)),
        sae_variant=SAEVariant.TOPK,
        top_k=(10,10,10,10,10),
        sae_keys=gen_sae_keys(n_features=5),
    ),
    SAEConfig(
        name="jumprelu-x8.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 8, 8, 8, 8)),
        sae_variant=SAEVariant.JUMP_RELU,
        sae_keys=gen_sae_keys(n_features=5),
    ),
    SAEConfig(
        name="jumprelu-staircase-x8.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 16, 24, 32, 40)),
        sae_variant=SAEVariant.JUMP_RELU_STAIRCASE,
        sae_keys=gen_sae_keys(n_features=5),
    ),
    SAEConfig(
        name="jumprelu-x16.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        # Only the penultimate layer needs the full x16 expansion factor
        n_features=tuple(64 * n for n in (4, 4, 4, 16, 4)),
        sae_variant=SAEVariant.JUMP_RELU,
        sae_keys=gen_sae_keys(n_features=5),
    ),
    SAEConfig(
        name="jumprelu-x32.stories_256x4",
        gpt_config=gpt_options["tiktoken_256x4"],
        n_features=tuple(256 * n for n in (32, 32, 32, 32, 32)),
        sae_variant=SAEVariant.JUMP_RELU,
        sae_keys=gen_sae_keys(n_features=5),
    ),
)
