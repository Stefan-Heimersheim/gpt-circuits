# %%
"""
Example of how to load a model and use it to generate text.
"""

import os
# %%
while not os.getcwd().endswith("gpt-circuits"):
    os.chdir("..")
print(os.getcwd())

import torch
from config.sae.training import LossCoefficients
from models.factorysparsified import FactorySparsified
from data.tokenizers import ASCIITokenizer

from david.utils import generate_with_saes

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

gpt_mlp = FactorySparsified.load("checkpoints/topk-staircase-share.shakespeare_64x4", device=device)
#gpt_mlp = FactorySparsified.load("checkpoints/jblock.shk_64x4-sparse-2.2e-04", device=device)
#gpt_mlp = FactorySparsified.load("checkpoints/jsae.shakespeare_64x4-sparsity-3.3e-02", device=device)
#gpt_mlp = FactorySparsified.load("checkpoints/topk-staircase-share.shakespeare_64x4", device=device)

gpt_mlp.to(device)


# %%


prompt = "Second Servingman:\nI will not so"
tokenizer = ASCIITokenizer()

import torch

print(f"Using no SAEs")
print(generate_with_saes(gpt_mlp, tokenizer, prompt, max_length=50, 
                   activations_to_patch=[]))
print("-"*100)

print(f"Using only 2_act")
print(generate_with_saes(gpt_mlp, tokenizer, prompt, max_length=50, 
                   activations_to_patch=["2_act"]))
print("-"*100)

print(f"Using 2_act and 3_act")
print(generate_with_saes(gpt_mlp, tokenizer, prompt, max_length=50, 
                   activations_to_patch=["2_act", "3_act"]))
print("-"*100)

print(f"Using all SAEs") 
print(generate_with_saes(gpt_mlp, tokenizer, prompt, max_length=50, 
                   activations_to_patch=gpt_mlp.saes.keys()))


# %%