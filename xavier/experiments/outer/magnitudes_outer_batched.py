#!/usr/bin/env python3
# filepath: xavier/experiments/compute_downstream_magnitudes.py

import torch
import sys
import os
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
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Imports from the project
from config.sae.models import SAEConfig
from models.sparsified import SparsifiedGPT
from models.gpt import GPT
from circuits import Circuit, Edge, Node, TokenlessNode, TokenlessEdge
from circuits.features.cache import ModelCache
from circuits.features.profiles import ModelProfile
from circuits.search.ablation import ResampleAblator
from circuits.search.edges import compute_batched_downstream_magnitudes_from_edges
from xavier.experiments import ExperimentParams, ExperimentResults, ExperimentOutput
from xavier.utils import create_tokenless_edges_from_array, get_attribution_rankings


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Compute downstream magnitudes & logits from edges")
    parser.add_argument("--num-edges", type=int, default=100, help="Number of edges to use")
    parser.add_argument("--upstream-layer-num", type=int, default=0, help="Upstream layer index")
    parser.add_argument("--num-samples", type=int, default=2, help="Number of samples for patching")
    parser.add_argument("--num-prompts", type=int, default=1, help="Number of prompts to use from validation data")
    parser.add_argument("--edge-selection", type=str, default="random", 
                        choices=["random", "gradient", "gradient_reversed", "outer"], help="Edge selection strategy")
    parser.add_argument("--sae-variant", type=str, default="standard", 
                        choices=["standard", "topk", "topk-x40", "topk-staircase", "jumprelu", "regularized", "top5", "top20", "topk"], help="Type of SAE")
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
    seed = args.seed

    if edge_selection == "outer":
        assert upstream_layer_num==3, "Only layer 3 is supported for outer product"

    # Set random seed
    random.seed(seed)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup model paths
    checkpoint_dir = project_root / "checkpoints"
    gpt_dir = checkpoint_dir / "shakespeare_64x4"
    sae_dir = checkpoint_dir / f"{sae_variant}.shakespeare_64x4"
    # sae_dir = checkpoint_dir / f"{sae_variant}.shakespeare_64x4"
    data_dir = project_root / "data"
    
    # Load GPT model
    print("Loading GPT model...")
    gpt = GPT.load(gpt_dir, device=device)
    
    # Load SAE config
    print("Loading SAE configuration...")
    sae_config_dir = sae_dir / "sae.json"
    with open(sae_config_dir, "r") as f:
        meta = json.load(f)
    config = SAEConfig(**meta)
    config.gpt_config = gpt.config
    
    # Create SparsifiedGPT model
    print("Creating SparsifiedGPT model...")
    model = SparsifiedGPT(config)
    model.gpt = gpt
    
    # Load SAE weights
    print("Loading SAE weights...")
    for layer_name, module in model.saes.items():
        weights_path = os.path.join(sae_dir, f"sae.{layer_name}.safetensors")
        load_model(module, weights_path, device=device.type)
    
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
    input_ids = reshaped_val_tensor[:num_prompts, :].to(device)  # Also move to the correct device
    
    start_time = time.time()

    downstream_magnitudes_list = []
    predicted_logits_list = []
    kl_div_list = []

    for prompt_idx in range(num_prompts):

        print(f"Computing upstream & downstream magnitudes (full circuit)...")
        with torch.no_grad():    
            with model.use_saes(activations_to_patch=[upstream_layer_num, upstream_layer_num + 1]) as encoder_outputs:
                _ = model(input_ids[prompt_idx].unsqueeze(0))
                upstream_magnitudes = encoder_outputs[upstream_layer_num].feature_magnitudes
                downstream_magnitudes_full_circuit = encoder_outputs[upstream_layer_num + 1].feature_magnitudes
        
        # Create edges
        num_upstream_features = model.config.n_features[upstream_layer_num]
        num_downstream_features = model.config.n_features[upstream_layer_num + 1]
        
        print(f"Creating {num_edges} edges using {edge_selection} selection strategy...")
        
        # Create edge array based on selection strategy

        # # Only sensible for a target token (here the final token)
        # full_outer_tensor = torch.einsum('tf,g->tfg', upstream_magnitudes.squeeze(), downstream_magnitudes_full_circuit.squeeze()[-1])
        # outer_tensor = torch.mean(full_outer_tensor, dim=0)
        # all_edges, _ = get_attribution_rankings(outer_tensor)
        # edge_arr = all_edges[:num_edges]

        # Only sensible for a target token (here the first token)
        outer_tensor = torch.einsum('f,g->fg', upstream_magnitudes.squeeze()[0], downstream_magnitudes_full_circuit.squeeze()[0])
        all_edges, _ = get_attribution_rankings(outer_tensor)
        edge_arr = all_edges[:num_edges]
        
        # Create TokenlessEdge objects
        edges = create_tokenless_edges_from_array(edge_arr, upstream_layer_num)
        
        # Compute downstream magnitudes from edges
        print(f"Computing downstream magnitudes from {len(edges)} edges...")
        
        if num_edges == num_upstream_features * num_downstream_features:
            # Use the full circuit magnitudes
            print(f"Using full circuit magnitudes...")
            downstream_magnitudes = downstream_magnitudes_full_circuit
        else:
            downstream_magnitudes = compute_batched_downstream_magnitudes_from_edges(
                model=model,
                ablator=ablator,
                edges=edges,
                upstream_magnitudes=upstream_magnitudes,
                target_token_idx=target_token_idx,
                num_samples=num_samples
            )

        # Compute logits subcircuit
        x_reconstructed = model.saes[str(upstream_layer_num + 1)].decode(downstream_magnitudes) 
        predicted_logits = model.gpt.forward_with_patched_activations(
            x_reconstructed, layer_idx=upstream_layer_num + 1
        )  # Shape: (num_batches, T, V)

        # Compute logits full circuit
        x_reconstructed_full_circuit = model.saes[str(upstream_layer_num + 1)].decode(downstream_magnitudes_full_circuit) 
        predicted_logits_full_circuit = model.gpt.forward_with_patched_activations(
            x_reconstructed_full_circuit, layer_idx=upstream_layer_num + 1
        )  # Shape: (num_batches, T, V)

        # Compute KL divergence between full and subcircuit logits
        probs_full_circuit = F.softmax(predicted_logits_full_circuit, dim=-1)
        probs = F.softmax(predicted_logits, dim=-1)

        # Compute KL divergence: KL(P||Q) = sum_i P(i) * log(P(i)/Q(i))
        epsilon = 1e-8
        kl_div = probs_full_circuit * torch.log((probs_full_circuit + epsilon) / (probs + epsilon))
        kl_div = kl_div.sum(dim=-1)  # Sum over vocabulary dimension

        print(f"KL divergence shape: {kl_div.shape}")

        downstream_magnitudes_list.append(downstream_magnitudes)
        predicted_logits_list.append(predicted_logits)
        kl_div_list.append(kl_div)

    all_downstream_magnitudes = torch.cat(downstream_magnitudes_list, dim=0)
    all_predicted_logits = torch.cat(predicted_logits_list, dim=0)
    all_kl_div = torch.cat(kl_div_list, dim=0)
    
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
        feature_magnitudes=all_downstream_magnitudes,
        logits=all_predicted_logits,
        kl_divergence=all_kl_div,
        execution_time=execution_time
    )
    
    experiment_output = ExperimentOutput(
        experiment_id=f"{experiment_params.task}_{sae_variant}_{edge_selection}_{upstream_layer_num}_{num_edges}",
        timestamp=datetime.datetime.now(),
        model_config=config,
        experiment_params=experiment_params,
        results=experiment_results
    )
 
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = project_root / f"xavier/experiments/data/run_16/{experiment_output.experiment_id}_{timestamp}.safetensors"
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