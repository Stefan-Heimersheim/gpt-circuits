from dataclasses import dataclass
from enum import Enum

from config import Config, map_options
from data.tokenizers import ASCIITokenizer, TikTokenTokenizer, Tokenizer


class NormalizationStrategy(str, Enum):
    LAYER_NORM = "LayerNorm"
    DYNAMIC_TANH = "DynamicTanh"
    IDENTITY = "Identity"


@dataclass
class GPTConfig(Config):
    block_size: int = 0  # max sequence length
    vocab_size: int = 0  # number of tokens
    n_layer: int = 0  # number of layers
    n_head: int = 0  # number of heads
    n_embd: int = 0  # embedding dimension
    norm_strategy: NormalizationStrategy = NormalizationStrategy.LAYER_NORM
    alpha_attn: float = 2.0 # DyT, only used during training
    alpha_mlp: float = 2.0 # DyT, only used during training

    @property
    def tokenizer(self) -> Tokenizer:
        """
        Infer tokenizer from vocabulary size.
        """
        match self.vocab_size:
            case TikTokenTokenizer.vocab_size:
                return TikTokenTokenizer()
            case ASCIITokenizer.vocab_size:
                return ASCIITokenizer()
            case _:
                raise ValueError(f"Unrecognized vocab size: {self.vocab_size}")

    @staticmethod
    def dict_factory(fields: list) -> dict:
        """
        Only export integer fields (exclude name and device)
        """
        # TODO: Is dangerous, should explicitly whitelist fields
        white_list = ['block_size', 'vocab_size', 'n_layer', 'n_head', 'n_embd', 'norm_strategy']
        return {k: v for (k, v) in fields if k in white_list}
    

# GPT configuration options
gpt_options: dict[str, GPTConfig] = map_options(
    GPTConfig(
        name="ascii_64x4",
        block_size=128,
        vocab_size=ASCIITokenizer.vocab_size,
        n_layer=4,
        n_head=4,
        n_embd=64,
    ),
    GPTConfig(
        name="gpt2",
        block_size=1024,
        vocab_size=50257,
        n_layer=12,
        n_head=12,
        n_embd=768
    ),
    GPTConfig(
        name="ascii_64x4_dyt",
        block_size=128,
        vocab_size=ASCIITokenizer.vocab_size,
        n_layer=4,
        n_head=4,
        n_embd=64,
        norm_strategy=NormalizationStrategy.DYNAMIC_TANH,
        alpha_mlp=10.0,
    ),
    GPTConfig(
        name="ascii_128x6",
        block_size=128,
        vocab_size=ASCIITokenizer.vocab_size,
        n_layer=6,
        n_head=4,
        n_embd=128,
    ),
    GPTConfig(
        name="tiktoken_32x4",
        block_size=128,
        vocab_size=TikTokenTokenizer.vocab_size,
        n_layer=4,
        n_head=16,
        n_embd=32,
    ),
    GPTConfig(
        name="tiktoken_256x4",
        block_size=128,
        vocab_size=TikTokenTokenizer.vocab_size,
        n_layer=4,
        n_head=16,
        n_embd=256,
    ),
)#type: ignore
