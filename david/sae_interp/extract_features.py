#!/usr/bin/env python3
"""
Generic SAE feature extraction script.
Extract feature magnitudes and indices for specific layers or all layers of a sparsified model.

Usage examples:
  # Process a specific layer
  python interp_sae.py --config staircase.tblock.gpt2.k32.x32 --layers 6
  
  # Process multiple specific layers
  python interp_sae.py --config staircase.tblock.gpt2.k32.x32 --layers 6 11 8
  
  # Process all available layers
  python interp_sae.py --config staircase.tblock.gpt2.k32.x32
  
  # Short form with custom settings
  python interp_sae.py -c my.config -l 6 8 11 --batch-size 4 --max-batches 1000
"""

import argparse
import sys
import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append('/workspace/gpt-circuits/')

from config.sae.models import sae_options
from models.factorysparsified import FactorySparsified

torch.set_grad_enabled(False)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract SAE features for specific layers or all layers')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='SAE config name (e.g., staircase.tblock.gpt2.k32.x32)')
    parser.add_argument('--layers', '-l', type=int, nargs='*', default=None,
                        help='Layer numbers to extract features from (e.g., -l 6 11 for layers 6 and 11, or no argument for all layers)')
    parser.add_argument('--data-dir', type=str, 
                        default='/workspace/gpt-circuits/data/fineweb_edu_10b/val_chunked_1024.npy',
                        help='Path to chunked token data')
    parser.add_argument('--output-dir', type=str,
                        default='/workspace/gpt-circuits/data/interp',
                        help='Base output directory for results')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for data loader')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Maximum number of batches to process (None for all)')
    return parser.parse_args()


def load_model(config_name, device):
    """Load the sparsified model."""
    print(f"Loading model with config: {config_name}")
    config = sae_options[config_name]
    
    # Construct model path - assume it matches the config name
    model_path = os.path.join("/workspace/gpt-circuits/checkpoints", config_name)
    model = FactorySparsified.load(model_path, device=config.device)
    model.to(config.device)
    
    print("Compiling model for better performance...")
    model = torch.compile(model)  # type: ignore
    torch.set_float32_matmul_precision('high')
    
    return model, config


def load_data(data_path, batch_size):
    """Load and prepare the data."""
    print(f"Loading data from: {data_path}")
    chunked_tokens_np = np.load(data_path, allow_pickle=False)
    
    # Convert to int32 if needed (PyTorch doesn't support uint16)
    if chunked_tokens_np.dtype == np.uint16:
        print(f"Converting data from {chunked_tokens_np.dtype} to int32")
        chunked_tokens_np = chunked_tokens_np.astype(np.int32)
    
    chunked_tokens = torch.tensor(chunked_tokens_np, dtype=torch.long)
    
    # Create TensorDataset and DataLoader
    val_dataset = TensorDataset(chunked_tokens)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Loaded {len(chunked_tokens)} sequences of length {chunked_tokens.shape[1]}")
    return val_dataloader


def get_available_layers(model):
    """Get all available SAE layers from the model."""
    # Extract layer numbers from SAE keys (assumes format like "6_act")
    layers = []
    for sae_key in model.saes.keys():
        if sae_key.endswith('_act'):
            layer_num = int(sae_key.split('_')[0])
            layers.append(layer_num)
    
    return sorted(layers)


def extract_multiple_features(model, config, dataloader, layers, max_batches=None):
    """Extract feature magnitudes and indices for multiple specified layers in a single pass."""
    print(f"Extracting features for {len(layers)} layers in a single pass: {layers}")
    
    # Verify all layers exist and create SAE keys list
    sae_keys_to_run = []
    for layer in layers:
        sae_layer = f'{layer}_act'
        if sae_layer not in model.saes:
            raise ValueError(f"SAE layer '{sae_layer}' not found in model. Available layers: {list(model.saes.keys())}")
        sae_keys_to_run.append(sae_layer)
    
    # Initialize storage for all layers
    all_feat_mags = {layer: [] for layer in layers}
    all_indices = {layer: [] for layer in layers}
    
    # Set up progress bar
    total_batches = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
    
    for i, x in enumerate(tqdm(dataloader, total=total_batches, desc=f"Processing {len(layers)} layers")):
        if max_batches is not None and i >= max_batches:
            break
            
        x = x[0]
        x = x.to(config.device)
        
        # Single forward pass with only the specified SAEs (more efficient!)
        out = model(x, stop_at_layer=max(layers)+1, sae_keys_to_run=sae_keys_to_run)
        
        # Extract features for each layer from the single output
        for layer in layers:
            sae_layer = f'{layer}_act'
            if sae_layer in out.indices and sae_layer in out.feature_magnitudes:
                idx = out.indices[sae_layer]
                feat_mag = out.feature_magnitudes[sae_layer]
                feat_mag_topk = torch.gather(feat_mag, dim=-1, index=idx)
                
                all_feat_mags[layer].append(feat_mag_topk.cpu().numpy())
                all_indices[layer].append(idx.cpu().numpy())
    
    # Concatenate results for each layer
    results = {}
    for layer in layers:
        feat_mags = np.concatenate(all_feat_mags[layer], axis=0)
        indices = np.concatenate(all_indices[layer], axis=0)
        results[layer] = (feat_mags, indices)
        print(f"Layer {layer} - Features shape: {feat_mags.shape}, Indices shape: {indices.shape}")
    
    return results


def extract_features(model, config, dataloader, layer, max_batches=None):
    """Extract feature magnitudes and indices for the specified layer."""
    sae_layer = f'{layer}_act'
    
    # Check if the SAE layer exists
    if sae_layer not in model.saes:
        raise ValueError(f"SAE layer '{sae_layer}' not found in model. Available layers: {list(model.saes.keys())}")
    
    feature_size = model.saes[sae_layer].feature_size
    print(f"Extracting features for layer {layer} (SAE key: {sae_layer})")
    print(f"Feature size: {feature_size}")
    
    feat_mags = []
    indices = []
    
    # Set up progress bar
    total_batches = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
    
    for i, x in enumerate(tqdm(dataloader, total=total_batches, desc=f"Processing layer {layer}")):
        if max_batches is not None and i >= max_batches:
            break
            
        x = x[0]
        x = x.to(config.device)
        
        # Run forward pass with only the specified SAE layer
        out = model(x, stop_at_layer=layer+1, sae_keys_to_run=[sae_layer])
        
        # Extract features
        idx = out.indices[sae_layer]
        feat_mag = out.feature_magnitudes[sae_layer]
        feat_mag_topk = torch.gather(feat_mag, dim=-1, index=idx)
        
        # Store results
        feat_mags.append(feat_mag_topk.cpu().numpy())
        indices.append(idx.cpu().numpy())
    
    # Concatenate all results
    feat_mags = np.concatenate(feat_mags, axis=0)
    indices = np.concatenate(indices, axis=0)
    
    print(f"Extracted features shape: {feat_mags.shape}")
    print(f"Extracted indices shape: {indices.shape}")
    
    return feat_mags, indices


def save_results(feat_mags, indices, output_dir, config_name, layer):
    """Save the extracted features to disk."""
    # Create output directory structure
    model_dir = Path(output_dir) / config_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save files
    feat_mags_path = model_dir / f"layer_{layer}_feat_mags.npy"
    indices_path = model_dir / f"layer_{layer}_indices.npy"
    
    feat_mags_optimized = feat_mags.astype(np.float16)  # Half precision for magnitudes
    indices_optimized = indices.astype(np.uint32)       # uint32 for indices (0-50256 range)
    
    np.save(feat_mags_path, feat_mags_optimized)
    np.save(indices_path, indices_optimized)
    
    print(f"Saved feature magnitudes to: {feat_mags_path}")
    print(f"Saved indices to: {indices_path}")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("=" * 60)
    print("SAE Feature Extraction")
    print("=" * 60)
    print(f"Config: {args.config}")
    if args.layers is not None:
        print(f"Layers: {args.layers}")
    else:
        print("Layers: ALL (will process all available layers)")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    try:
        # Load model
        model, config = load_model(args.config, 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load data
        dataloader = load_data(args.data_dir, args.batch_size)
        
        # Determine which layers to process
        if args.layers is not None:
            layers_to_process = args.layers
            # Validate that all specified layers exist
            available_layers = get_available_layers(model)
            invalid_layers = [layer for layer in layers_to_process if layer not in available_layers]
            if invalid_layers:
                raise ValueError(f"Invalid layer(s): {invalid_layers}. Available layers: {available_layers}")
            if len(layers_to_process) > 1:
                print(f"Processing {len(layers_to_process)} specified layers: {layers_to_process}")
                print("-" * 60)
        else:
            layers_to_process = get_available_layers(model)
            print(f"Found {len(layers_to_process)} SAE layers: {layers_to_process}")
            print("-" * 60)
        
        # Process layers efficiently
        if len(layers_to_process) == 1:
            # Single layer - use targeted approach (only runs specific SAE)
            layer = layers_to_process[0]
            feat_mags, indices = extract_features(model, config, dataloader, layer, args.max_batches)
            save_results(feat_mags, indices, args.output_dir, args.config, layer)
        else:
            # Multiple layers - use efficient single-pass approach (runs all SAEs once)
            # This is much faster than doing separate forward passes for each layer
            all_results = extract_multiple_features(model, config, dataloader, layers_to_process, args.max_batches)
            
            print("-" * 60)
            print(f"Saving results for {len(layers_to_process)} layers...")
            
            # Save results for each layer
            for layer in layers_to_process:
                feat_mags, indices = all_results[layer]
                save_results(feat_mags, indices, args.output_dir, args.config, layer)
                print(f"Saved results for layer {layer}")
        
        print("=" * 60)
        if len(layers_to_process) == 1:
            print("Feature extraction completed successfully!")
        else:
            print(f"Feature extraction completed successfully for all {len(layers_to_process)} layers!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
