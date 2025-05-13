# %%
#!/usr/bin/env python3
# filepath: xavier/experiments/compute_downstream_magnitudes.py
import os

# Get current directory and keep going up until we find gpt-circuits root
while not os.getcwd().endswith("gpt-circuits"):
    os.chdir("..")
print(os.getcwd())

# %%


import torch
import sys
from pathlib import Path
import time
import datetime
import random
import numpy as np
import json
import argparse
from safetensors.torch import load_model, load_file, save_file
import torch.nn.functional as F

# Path setup
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
print(project_root)

# Imports from the project
from config.sae.models import SAEConfig
from models.sparsified import SparsifiedGPT
from models.gpt import GPT
from models.factorysparsified import FactorySparsified
from circuits import Circuit, Edge, Node, TokenlessNode, TokenlessEdge
from circuits.features.cache import ModelCache
from circuits.features.profiles import ModelProfile
from circuits.search.ablation import ResampleAblator
from circuits.search.edges import compute_batched_downstream_magnitudes_from_edges
from xavier.experiments import ExperimentParams, ExperimentResults, ExperimentOutput
from xavier.utils import create_tokenless_edges_from_array, get_attribution_rankings

# %%

# Experiment parameters
num_edges = 10
upstream_layer_num = 1
num_samples = 2
num_prompts = 3
edge_selection = "random"
sae_variant = "topk-staircase-share"
seed = 125

torch.manual_seed(seed)
random.seed(seed)

# Force CPU for consistent device usage
device = torch.device("cpu")
print(f"Using device: {device}")

# Setup model paths
checkpoint_dir = project_root / "checkpoints"
gpt_dir = checkpoint_dir / "shakespeare_64x4"
data_dir = project_root / "data"
mlp_dir = checkpoint_dir / f"{sae_variant}.shakespeare_64x4"

# Load GPT model
print("Loading GPT model...")

model = FactorySparsified.load(mlp_dir, device=device)
model.to(device)

# Load SAE config
meta_path = os.path.join(mlp_dir, "sae.json")
with open(meta_path, "r") as f:
    meta = json.load(f)
config = SAEConfig(**meta)

# Create a model profile (will only work for zero ablation)
model_profile = ModelProfile()

# Create an empty cache (since we won't use it for zero ablation)
model_cache = ModelCache()

# Create the ResampleAblator with k_nearest=0 for zero ablation
ablator = ResampleAblator(
    model_profile=model_profile,
    model_cache=model_cache,
    k_nearest=0  # This setting enables zero ablation
)

# Load validation data
val_data_dir = data_dir / 'shakespeare/val_000000.npy'
with open(val_data_dir, 'rb') as f:
    val_array = np.load(f)
val_tensor = torch.from_numpy(val_array).to(torch.long)  # Convert to long tensor

# Calculate number of complete chunks we can make
sequence_length = 128 # Hardcoded for now
target_token_idx = sequence_length - 1
print('hello?')
num_chunks = val_tensor.shape[0] // sequence_length
usable_data = val_tensor[:num_chunks * sequence_length]
reshaped_val_tensor = usable_data.reshape(num_chunks, sequence_length)
input_ids = reshaped_val_tensor[:num_prompts, :].to(device)  # Also move to the correct device

print(f"Computing upstream & downstream magnitudes (full circuit)...")
with torch.no_grad():    
    with model.use_saes(activations_to_patch=[upstream_layer_num, upstream_layer_num + 1]) as encoder_outputs:
        _ = model(input_ids)
        upstream_magnitudes = encoder_outputs[f'{upstream_layer_num}'].feature_magnitudes
        downstream_magnitudes_full_circuit = encoder_outputs[f'{upstream_layer_num + 1}'].feature_magnitudes

print(upstream_magnitudes)
print(downstream_magnitudes_full_circuit)

# %%