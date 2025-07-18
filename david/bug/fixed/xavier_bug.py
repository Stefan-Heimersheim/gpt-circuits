# %%
import os

# Path setup
# Get current directory and keep going up until we find gpt-circuits root
while not os.getcwd().endswith("gpt-circuits"):
    os.chdir("..")
print(os.getcwd())

import sys  # Add this import
import torch
import random
import time
import datetime
from pathlib import Path
import numpy as np
import json
import argparse
from transformer_lens.hook_points import HookPoint
import torch.nn.functional as F
from safetensors.torch import load_model, load_file, save_file

# Path setup
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


from config.sae.models import SAEConfig, SAEVariant
from config.sae.training import LossCoefficients
from models.sae.topk import StaircaseTopKSAE
from models.gpt import GPT
from models.factorysparsified import FactorySparsified
from models.mlpsparsified import MLPSparsifiedGPT
from data.tokenizers import ASCIITokenizer
from david.convert_to_tl import convert_gpt_to_transformer_lens
from david.convert_to_tl import run_tests as run_tl_tests
from xavier.utils import create_tokenless_edges_from_array, get_attribution_rankings

from circuits import Circuit
from circuits.search.divergence import (
    compute_downstream_magnitudes_mlp,
    patch_feature_magnitudes,
)
from circuits.features.cache import ModelCache
from circuits.features.profiles import ModelProfile
from circuits.search.ablation import ResampleAblator
from circuits.search.edges import compute_batched_downstream_magnitudes_from_edges_mlp, compute_batched_downstream_magnitudes_from_edges_resid
from xavier.experiments import ExperimentParams, ExperimentResults, ExperimentOutput

@torch.no_grad()
def main():

    # Experiment parameters
    num_edges = 10
    upstream_layer_num = 1
    num_samples = 2
    num_prompts = 3
    edge_selection = "random"
    sae_variant = "staircase"
    seed = 125

    torch.manual_seed(seed)

    # Set random seed
    random.seed(seed)

    # Device setup
    device = torch.device("cpu") #torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Setup model paths
    checkpoint_dir = project_root / "checkpoints"
    gpt_dir = checkpoint_dir / "shakespeare_64x4"
    data_dir = project_root / "data"

    # Hacky fix for loading the model
    # TODO: Use SAEVariant instead of strings. -David
    # from config.sae.models import SAEVariant
    if sae_variant == "mlp-topk":
        mlp_dir = checkpoint_dir / f"{sae_variant}.shakespeare_64x4"
    elif sae_variant == "jsae":
        mlp_dir = checkpoint_dir / f"{sae_variant}.shakespeare_64x4"
    elif sae_variant == "staircase":
        mlp_dir = checkpoint_dir / "staircase-mlpblock.shk_64x4"
    else:
        mlp_dir = checkpoint_dir / f"jblock.shk_64x4-sparse-{sae_variant}"

    # Load GPT model
    print("Loading GPT model...")

    model = FactorySparsified.load(mlp_dir)
    model.to(device)


    # not a great fix, but it will do for now
    for key, sae in model.saes.items():
        model.saes[key].shared_context.W_dec = sae.shared_context.W_dec.to(device)
        model.saes[key].shared_context.W_enc = sae.shared_context.W_enc.to(device)
        model.saes[key].W_dec = sae.W_dec.to(device)
        model.saes[key].W_enc = sae.W_enc.to(device)


   
    for param in model.parameters():
        param.data = param.data.to(device)


    # Load validation data
    val_data_dir = data_dir / 'shakespeare/val_000000.npy'
    with open(val_data_dir, 'rb') as f:
        val_array = np.load(f)
    val_tensor = torch.from_numpy(val_array).to(torch.long)  # Convert to long tensor
 
    # Calculate number of complete chunks we can make
    sequence_length = 128 # Hardcoded for now
    target_token_idx = sequence_length - 1

    num_chunks = val_tensor.shape[0] // sequence_length
    usable_data = val_tensor[:num_chunks * sequence_length]
    reshaped_val_tensor = usable_data.reshape(num_chunks, sequence_length)
    input_ids = reshaped_val_tensor[:num_prompts, :].to(device)  # Also move to the correct device

    print(f"Computing upstream & downstream magnitudes (full circuit)...")
    keys = [f'{upstream_layer_num}_residmid', f'{upstream_layer_num}_residpost']
    with model.use_saes(activations_to_patch = keys) as encoder_outputs:
        _ = model(input_ids.to(device))
        upstream_magnitudes = encoder_outputs[keys[0]].feature_magnitudes
        downstream_magnitudes_full_circuit = encoder_outputs[keys[1]].feature_magnitudes

    print("Done!")



if __name__ == "__main__":
    main()
# %%
