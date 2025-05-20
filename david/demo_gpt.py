# %%
%load_ext autoreload
%autoreload 2
# %%
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_grad_enabled(False)

# Path setup
# Get current directory and keep going up until we find gpt-circuits root
while not os.getcwd().endswith("gpt-circuits"):
    os.chdir("..")
print(os.getcwd())

# %%
from models.gpt import GPT
from models.sparsified import SparsifiedGPT
from safetensors.torch import load_model
from data.tokenizers import ASCIITokenizer, TikTokenTokenizer
from config.sae.models import SAEConfig
from david.utils import gpt_generate
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_gpt_model():
    gpt_dir = Path("checkpoints/tiny_32x4")
    gpt = GPT.load(gpt_dir, device=device)
    tokenizer = TikTokenTokenizer()
    return gpt, tokenizer


# def load_and_prepare_data_loader(device):
#     val_activations_path = "data/shakespeare/val_000000.npy"
#     val_activations = np.load(val_activations_path, allow_pickle=False).astype(np.int32)
#     val_activations = torch.tensor(val_activations, device=device)
    
#     chunk_size = 128
#     batch_size = 32
#     N = val_activations.shape[0]
#     val_activations = val_activations[:(N//chunk_size)*chunk_size].long()
#     val_activations = val_activations.view(-1, chunk_size)
#     val_loader = torch.utils.data.DataLoader(val_activations, batch_size=batch_size, shuffle=False)
    
#     return val_loader

gpt, tokenizer = load_gpt_model()
print(gpt_generate(gpt, tokenizer, "Today I thought,", max_length=100))
# %%