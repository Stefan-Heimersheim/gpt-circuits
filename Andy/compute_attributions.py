from attributor_new import IntegratedGradientAttributor, ManualAblationAttributor, PathType
import os
import json
import sys
sys.path.append('/workspace/gpt-circuits')

import argparse
from pathlib import Path

from utils import sorted_indices_by_value

from config.sae.training import options
from config.sae.models import sae_options, SAEVariant
from data.dataloaders import TrainingDataLoader
import torch
from models.factorysparsified import FactorySparsified

from models.gpt import GPT
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput


from safetensors.torch import load_file, save_file

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--load_from", type=str, help="Model weights to load")
    parser.add_argument("--save_to", type=str, help="Path to save the attributions")
    parser.add_argument("--data_dir", type=str, help="Directory containing the data")
    #parser.add_argument("--path_type", type = str, help="BLOCK, MLP, or MLP_LAYER")

    parser.add_argument("--attribution_method", type=str, choices=["ig", "ma"], default = 'ig', help="Attribution method to use, either 'ig' or 'ma'")

    parser.add_argument("--save_name", type=str, default = '', help="Name of experiment to save")
    
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loading")
    parser.add_argument("--num_batches", type=int, default=32, help="Number of batches to process")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps in ig path (ig only)")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Epsilon value for ma (ma only)")
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose output")
    parser.add_argument("--layers", type=str, default='all', help="Layers to compute attributions for, either 'all' or a comma-separated list of layer indices")
    #parser.add_argument("--config", type="", help="Model config")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #print(device)
    model_path = Path(args.load_from)
    model = FactorySparsified.load(args.load_from, device=device)
    model.to(device)
    
    data_dir = args.data_dir
    batch_size = args.batch_size

    if args.layers == 'all':
        layers = list(range(model.config.n_layers))
    else:
        layers = [int(layer) for layer in args.layers.split(',')]


    dataloader = TrainingDataLoader(
        dir_path=data_dir,
        B= batch_size,
        T=model.config.block_size,
        process_rank=0,
        num_processes=1,
        split="val",
    )

    if args.attribution_method == "ig":
        attributor = IntegratedGradientAttributor(model, dataloader, nbatches = args.num_batches, verbose=args.verbose, steps=args.steps, chunk_size=4)
        attributions = attributor.layer_by_layer(layers=layers)
    elif args.attribution_method == "ma":
        attributor = ManualAblationAttributor(model, dataloader, nbatches = args.num_batches, verbose=args.verbose, epsilon=args.epsilon)
        attributions = attributor.layer_by_layer(layers=layers)

    #attributions_listed = {}
    #for key in attributions.keys():
        #attributions_listed[key] = sorted_indices_by_value(attributions[key])


    output_filename = args.save_to
    name = args.save_name
    if name == '':
        name = string(os.path.basename(os.path.normpath(model_path))) + '_' + args.attribution_method

    if output_filename.endswith(".safetensors"):
        path = Path(output_filename)
    else:
        path = os.path.join(output_filename, f"{name}.safetensors")

    save_file(
        attributions,
        path,
        metadata={
            "model_name": model_path.name,
            "attribution_method": args.attribution_method,
            "batch_size": f"{batch_size}",
            "num_batches": f"{args.num_batches}",
            "steps": f"{args.steps}" if args.attribution_method == "ig" else "N/A",
            "epsilon": f"{args.epsilon}" if args.attribution_method == "ma" else "N/A",
        },
    )

    print(f"Attributions saved to {name} in {path}")
    





    