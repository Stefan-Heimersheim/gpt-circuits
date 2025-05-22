import os
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
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.sae.models import SAEConfig, SAEVariant
from config.sae.training import LossCoefficients
from models.sae.topk import StaircaseTopKSAE
from models.gpt import GPT
from models.factorysparsified import FactorySparsified
from models.mlpsparsified import MLPSparsifiedGPT
from data.tokenizers import ASCIITokenizer
from demo.convert_to_tl import convert_gpt_to_transformer_lens
from demo.convert_to_tl import run_tests as run_tl_tests
from ablation.utils import create_tokenless_edges_from_array, get_attribution_rankings

from circuits import Circuit
from circuits.search.divergence import (
    compute_downstream_magnitudes_mlp,
    patch_feature_magnitudes,
)
from circuits.features.cache import ModelCache
from circuits.features.profiles import ModelProfile
from circuits.search.ablation import ResampleAblator
from circuits.search.edges import compute_batched_downstream_magnitudes_from_edges_mlp, compute_batched_downstream_magnitudes_from_edges_resid
from ablation import ExperimentParams, ExperimentResults, ExperimentOutput

@torch.no_grad()
def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Compute downstream magnitudes & logits from edges")
    parser.add_argument("--num-edges", type=int, default=100, help="Number of edges to use")
    parser.add_argument("--upstream-layer-num", type=int, default=0, help="Upstream layer index")
    parser.add_argument("--num-samples", type=int, default=2, help="Number of samples for patching")
    parser.add_argument("--num-prompts", type=int, default=1, help="Number of prompts to use from validation data")
    parser.add_argument("--edge-selection", type=str, default="random", choices=["random", "gradient"], help="Edge selection strategy")
    parser.add_argument("--sae-variant", type=str, default="standard", choices=["topk", "staircase"], help="Type of SAE")
    parser.add_argument("--run-index", type=str, default="testing", help="Index of the run")
    parser.add_argument("--seed", type=int, default=125, help="Random seed")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)

    # Experiment parameters
    num_edges = args.num_edges
    upstream_layer_num = args.upstream_layer_num
    num_samples = args.num_samples
    num_prompts = args.num_prompts
    edge_selection = args.edge_selection
    sae_variant = args.sae_variant
    run_idx = args.run_index
    seed = args.seed

    # Set random seed
    random.seed(seed)

    # Device setup
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Setup model paths
    mlp_dir = project_root / "checkpoints" / f"ff_block_{sae_variant}"
    data_dir = project_root / "data"

    # Load GPT model
    print("Loading GPT model...")

    model = FactorySparsified.load(mlp_dir, device=device)
    model.to(device)

    # Load SAE config
    meta_path = os.path.join(mlp_dir, "sae.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    config = SAEConfig(**meta)
    
    # config.gpt_config = gpt.config

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

    num_chunks = val_tensor.shape[0] // sequence_length
    usable_data = val_tensor[:num_chunks * sequence_length]
    reshaped_val_tensor = usable_data.reshape(num_chunks, sequence_length)
    input_ids = reshaped_val_tensor[:num_prompts, :].to(device) 

    print(f"Computing upstream & downstream magnitudes (full circuit)...")
    keys = [f'{upstream_layer_num}_residmid', f'{upstream_layer_num}_residpost']
    with model.use_saes(activations_to_patch = keys) as encoder_outputs:
        _ = model(input_ids)
        upstream_magnitudes = encoder_outputs[keys[0]].feature_magnitudes
        downstream_magnitudes_full_circuit = encoder_outputs[keys[1]].feature_magnitudes

    # Create edges
    num_upstream_features = model.config.n_features[upstream_layer_num]
    num_downstream_features = model.config.n_features[upstream_layer_num + 1]
    
    print(f"Creating {num_edges} edges using {edge_selection} selection strategy...")
    
    # Create edge array based on selection strategy
    if edge_selection == "random":
        # Create a random permutation of all possible edges
        all_edges = [(a, b) for a in range(num_upstream_features) for b in range(num_downstream_features)]
        random.shuffle(all_edges)
        edge_arr = all_edges[:num_edges]

    elif edge_selection == "gradient":
        gradient_dir = project_root / f"attributions/data/ff_block_{sae_variant}.safetensors"
        tensors = load_file(gradient_dir)
        all_edges, _ = get_attribution_rankings(tensors[f'MLP_BLOCK_{upstream_layer_num}'])
        edge_arr = all_edges[:num_edges]


    # Create TokenlessEdge objects
    edges = create_tokenless_edges_from_array(edge_arr, upstream_layer_num)
    
    # Compute downstream magnitudes from edges
    print(f"Computing downstream magnitudes from {len(edges)} edges...")
    start_time = time.time()
    
    print(upstream_magnitudes.shape)
    print(downstream_magnitudes_full_circuit.shape)
    if num_edges == num_upstream_features * num_downstream_features:
        # Use the full circuit magnitudes
        print(f"Using full circuit magnitudes...")
        downstream_magnitudes = downstream_magnitudes_full_circuit

    else:
        # Compute downstream magnitudes from edges
        downstream_magnitudes = compute_batched_downstream_magnitudes_from_edges_resid(
            model=model,
            ablator=ablator,
            edges=edges,
            upstream_magnitudes=upstream_magnitudes,
            target_token_idx=target_token_idx,
            num_samples=num_samples
        )

    # Compute logits subcircuit
    x_reconstructed = model.saes[f'{upstream_layer_num}_residpost'].decode(downstream_magnitudes) 
    predicted_logits = model.gpt.forward_with_patched_activations(
        x_reconstructed, upstream_layer_num + 1
    )   # Shape: (num_batches, T, V)
    print(predicted_logits.shape)

    # Compute logits full circuit
    x_reconstructed_full_circuit = model.saes[f'{upstream_layer_num}_residpost'].decode(downstream_magnitudes_full_circuit) 
    predicted_logits_full_circuit = model.gpt.forward_with_patched_activations(
        x_reconstructed_full_circuit, upstream_layer_num + 1
    )  # Shape: (num_batches, T, V)
    print(predicted_logits_full_circuit.shape)

    # Compute logits full model
    predicted_logits_full_model = model(input_ids, targets=None, is_eval=True).logits


    # Compute KL divergence between full and subcircuit logits
    probs_full_model = F.softmax(predicted_logits_full_model, dim=-1)
    probs_full_circuit = F.softmax(predicted_logits_full_circuit, dim=-1)
    probs = F.softmax(predicted_logits, dim=-1)

    # Compute KL divergence: KL(P||Q) = sum_i P(i) * log(P(i)/Q(i))
    epsilon = 1e-8
    kl_div = probs_full_circuit * torch.log((probs_full_circuit + epsilon) / (probs + epsilon))
    kl_div = kl_div.sum(dim=-1)  # Sum over vocabulary dimension

    kl_div_full_model = probs_full_model * torch.log((probs_full_model + epsilon) / (probs + epsilon))
    kl_div_full_model = kl_div_full_model.sum(dim=-1)  # Sum over vocabulary dimension

    print(f"KL divergence shape: {kl_div.shape}")

    # Compute the time taken for the computation
    execution_time = time.time() - start_time
    print(f"Computation completed in {execution_time:.2f} seconds")

    # Create experiment output
    experiment_params = ExperimentParams(
        task="magnitudes",
        ablator="zero",
        edges = edge_arr,
        edge_selection_strategy=edge_selection,
        num_edges=num_edges,
        upstream_layer_idx=upstream_layer_num,
        num_samples=num_samples,
        num_prompts=num_prompts,
        random_seed=seed,  
        dataset_name=None
    )
    
    experiment_results = ExperimentResults(
        feature_magnitudes=downstream_magnitudes,
        logits=predicted_logits,
        kl_divergence=kl_div,
        kl_divergence_full_model=kl_div_full_model,
        execution_time=execution_time
    )
    
    experiment_output = ExperimentOutput(
        experiment_id=f"{experiment_params.task}_{sae_variant}_{edge_selection}_{upstream_layer_num}_{num_edges}",
        timestamp=datetime.datetime.now(),
        model_config=config,
        experiment_params=experiment_params,
        results=experiment_results
    )
 
    # CHECK THIS 
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = project_root / f"ablation/data/{run_idx}"
    output_path = output_dir / f"{experiment_output.experiment_id}_{timestamp}.safetensors"
    if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
    experiment_output.to_safetensor(output_path)
    
    print("Done!")
    
    # Print summary
    print(f"\nSummary:")
    print(f"- Upstream layer: {args.upstream_layer_num}")
    print(f"- Number of edges: {args.num_edges}")
    print(f"- Downstream magnitudes shape: {downstream_magnitudes.shape}")
    print(f"- Downstream logits shape: {predicted_logits.shape}")
    print(f"- Results saved to: {output_path}")

if __name__ == "__main__":
    main()