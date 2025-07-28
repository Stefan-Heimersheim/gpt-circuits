import sys
sys.path.append('/workspace/gpt-circuits')

import torch as t
import torch.nn as nn
from torch import Tensor
from torch.autograd.functional import jacobian
import os
import time
from datetime import datetime, timedelta

from safetensors.torch import save_file

import einops

from enum import Enum

from utils import MaxSizeList, get_SAE_activations, SkipModule, PathType

from models.gpt import GPT
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput

from config.gpt.training import options
from config.sae.models import sae_options, SAEVariant

from data.tokenizers import ASCIITokenizer, TikTokenTokenizer

from models.sae import SparseAutoencoder
from typing import Callable, Optional

from data.dataloaders import TrainingDataLoader
TensorFunction = Callable[[Tensor], Tensor]

# Add support for DataParallel
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class Attributor():
    def __init__(self, model: nn.Module, dataloader: TrainingDataLoader, nbatches: int = 32, 
                 verbose: bool = False, save_dir: str = "./attributions", 
                 device_ids: Optional[list] = None, use_ddp: bool = False):
        """
        Returns a dict of all consecutive integrated gradient attributions for a model.
        :param model: SparsifiedGPT model
        :param dataloader: Dataloader
        :param nbatches: How many batches of data to aggregate into attributions
        :param verbose: Prints updates after finishing each layer connection
        :param save_dir: Directory to save intermediate results
        :param device_ids: List of GPU device IDs to use (e.g., [0, 1, 2, 3])
        :param use_ddp: Whether to use DistributedDataParallel (recommended for multi-node)
        """
        self.dataloader = dataloader
        self.nbatches = nbatches
        self.verbose = verbose
        self.save_dir = save_dir
        self.attributions = {}
        self.use_ddp = use_ddp
        self.timing_stats = {}
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Setup multi-GPU
        if device_ids is None:
            device_ids = list(range(t.cuda.device_count()))
        self.device_ids = device_ids
        
        if len(device_ids) > 1:
            if use_ddp:
                # Initialize DDP if not already initialized
                if not dist.is_initialized():
                    dist.init_process_group(backend='nccl')
                self.model = DDP(model, device_ids=device_ids)
                self.base_model = self.model.module
            else:
                # Use DataParallel for single-node multi-GPU
                self.model = DataParallel(model, device_ids=device_ids)
                self.base_model = self.model.module
        else:
            self.model = model
            self.base_model = model
            
        # This is a mess. Fix later
        if self.base_model.config.sae_variant in [SAEVariant.JSAE_BLOCK, SAEVariant.STAIRCASE_BLOCK]:
            self.paths = PathType.MLP_LAYER
        elif self.base_model.config.sae_variant in [SAEVariant.JSAE_LAYER, SAEVariant.JSAE]:
            self.paths = PathType.MLP
        else:  
            self.paths = PathType.BLOCK
        if verbose:
            print(f"{'='*60}")
            print(f"Attributor Initialization")
            print(f"{'='*60}")
            print(f"Path type: {self.paths}")
            print(f"Number of GPUs: {len(device_ids)}")
            print(f"GPU devices: {device_ids}")
            print(f"Using DDP: {use_ddp}")
            print(f"Save directory: {save_dir}")
            print(f"Number of batches: {nbatches}")
            print(f"{'='*60}\n")

    def log_time(self, operation: str, duration: float):
        """Log timing information"""
        if operation not in self.timing_stats:
            self.timing_stats[operation] = []
        self.timing_stats[operation].append(duration)
        
        if self.verbose:
            avg_time = sum(self.timing_stats[operation]) / len(self.timing_stats[operation])
            print(f"  ⏱️  {operation}: {duration:.2f}s (avg: {avg_time:.2f}s)")

    def save_layer_attribution(self, layer_name: str, attribution: Tensor):
        """Save attribution for a single layer"""
        start_time = time.time()
        save_path = os.path.join(self.save_dir, f"attribution_{layer_name}.pt")
        t.save(attribution, save_path)
        save_time = time.time() - start_time
        
        if self.verbose:
            print(f"\n💾 Saved attribution for {layer_name}")
            print(f"   Path: {save_path}")
            print(f"   Shape: {attribution.shape}")
            print(f"   Time: {save_time:.2f}s")
    
    def load_existing_attributions(self):
        """Load any existing attributions from save directory"""
        if self.verbose:
            print(f"\n🔍 Checking for existing attributions in {self.save_dir}...")
        
        existing_files = [f for f in os.listdir(self.save_dir) if f.startswith("attribution_") and f.endswith(".pt")]
        
        if existing_files:
            print(f"📁 Found {len(existing_files)} existing attribution files")
            
        for file in existing_files:
            layer_name = file.replace("attribution_", "").replace(".pt", "")
            start_time = time.time()
            self.attributions[layer_name] = t.load(os.path.join(self.save_dir, file))
            load_time = time.time() - start_time
            if self.verbose:
                print(f"   ✓ Loaded {layer_name} ({load_time:.2f}s)")

    def layer_by_layer(self, layers: list[int] = []) -> dict:
        total_start_time = time.time()
        
        # Load any existing attributions
        self.load_existing_attributions()
        
        if len(layers) == 0:
            layers = list(range(self.base_model.gpt.config.n_layer))
        else:
            layers = layers
        assert len(layers) > 0, "Layers must be a non-empty list"
        assert min(layers) >= 0, "Layers must be a non-negative list"
        assert max(layers) < self.base_model.gpt.config.n_layer, "Layers must be less than the number of layers in the model"
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting layer-by-layer attribution")
            print(f"Total layers to process: {len(layers)}")
            print(f"Layers: {layers}")
            print(f"{'='*60}\n")
        
        processed_count = 0
        skipped_count = 0
        
        if self.paths == PathType.BLOCK:
            for i in layers:
                layer_name = f'{i}-{i+1}'
                # Skip if already computed
                if layer_name in self.attributions:
                    if self.verbose:
                        print(f"\n⏭️  Skipping {layer_name} - already computed")
                    skipped_count += 1
                    continue
                
                if self.verbose:
                    print(f"\n{'='*50}")
                    print(f"🔄 Processing connections from Layer {i} to {i+1}")
                    print(f"   Progress: {processed_count + 1}/{len(layers) - skipped_count}")
                    print(f"{'='*50}")
                
                layer_start_time = time.time()
                attribution = self.single_layer(i, i+1)
                layer_time = time.time() - layer_start_time
                
                self.attributions[layer_name] = attribution
                self.save_layer_attribution(layer_name, attribution)
                self.dataloader.reset()
                
                processed_count += 1
                if self.verbose:
                    print(f"\n✅ Finished Layer {i} to {i+1}")
                    print(f"   Total layer time: {layer_time:.2f}s")
                    elapsed = time.time() - total_start_time
                    print(f"   Total elapsed: {str(timedelta(seconds=int(elapsed)))}")
                    remaining_layers = len(layers) - processed_count - skipped_count
                    if remaining_layers > 0 and processed_count > 0:
                        avg_time_per_layer = (elapsed - skipped_count * 0.1) / processed_count
                        est_remaining = avg_time_per_layer * remaining_layers
                        print(f"   Estimated remaining: {str(timedelta(seconds=int(est_remaining)))}")
                        
        elif self.paths == PathType.MLP:
            for i in layers:
                layer_name = f'MLP_{i}'
                # Skip if already computed
                if layer_name in self.attributions:
                    if self.verbose:
                        print(f"\n⏭️  Skipping {layer_name} - already computed")
                    skipped_count += 1
                    continue
                
                if self.verbose:
                    print(f"\n{'='*50}")
                    print(f"🔄 Processing MLP in Layer {i}")
                    print(f"   Progress: {processed_count + 1}/{len(layers) - skipped_count}")
                    print(f"{'='*50}")
                
                layer_start_time = time.time()
                attribution = self.single_layer(i)
                layer_time = time.time() - layer_start_time
                
                self.attributions[layer_name] = attribution
                self.save_layer_attribution(layer_name, attribution)
                self.dataloader.reset()
                
                processed_count += 1
                if self.verbose:
                    print(f"\n✅ Finished MLP in Layer {i}")
                    print(f"   Total layer time: {layer_time:.2f}s")
                    elapsed = time.time() - total_start_time
                    print(f"   Total elapsed: {str(timedelta(seconds=int(elapsed)))}")
                    remaining_layers = len(layers) - processed_count - skipped_count
                    if remaining_layers > 0 and processed_count > 0:
                        avg_time_per_layer = (elapsed - skipped_count * 0.1) / processed_count
                        est_remaining = avg_time_per_layer * remaining_layers
                        print(f"   Estimated remaining: {str(timedelta(seconds=int(est_remaining)))}")
                        
        elif self.paths == PathType.MLP_LAYER:
            for i in layers:
                layer_name = f'MLP_BLOCK_{i}'
                # Skip if already computed
                if layer_name in self.attributions:
                    if self.verbose:
                        print(f"\n⏭️  Skipping {layer_name} - already computed")
                    skipped_count += 1
                    continue
                
                if self.verbose:
                    print(f"\n{'='*50}")
                    print(f"🔄 Processing MLP block in Layer {i}")
                    print(f"   Progress: {processed_count + 1}/{len(layers) - skipped_count}")
                    print(f"{'='*50}")
                
                layer_start_time = time.time()
                attribution = self.single_layer(i)
                layer_time = time.time() - layer_start_time
                
                self.attributions[layer_name] = attribution
                self.save_layer_attribution(layer_name, attribution)
                self.dataloader.reset()
                
                processed_count += 1
                if self.verbose:
                    print(f"\n✅ Finished MLP block in Layer {i}")
                    print(f"   Total layer time: {layer_time:.2f}s")
                    elapsed = time.time() - total_start_time
                    print(f"   Total elapsed: {str(timedelta(seconds=int(elapsed)))}")
                    remaining_layers = len(layers) - processed_count - skipped_count
                    if remaining_layers > 0 and processed_count > 0:
                        avg_time_per_layer = (elapsed - skipped_count * 0.1) / processed_count
                        est_remaining = avg_time_per_layer * remaining_layers
                        print(f"   Estimated remaining: {str(timedelta(seconds=int(est_remaining)))}")
        
        total_time = time.time() - total_start_time
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🎉 Attribution Complete!")
            print(f"   Total layers processed: {processed_count}")
            print(f"   Layers skipped (already computed): {skipped_count}")
            print(f"   Total time: {str(timedelta(seconds=int(total_time)))}")
            print(f"{'='*60}\n")
            
            # Print timing statistics
            if self.timing_stats:
                print(f"\n📊 Timing Statistics:")
                for op, times in self.timing_stats.items():
                    avg_time = sum(times) / len(times)
                    total_op_time = sum(times)
                    print(f"   {op}:")
                    print(f"      Average: {avg_time:.2f}s")
                    print(f"      Total: {total_op_time:.2f}s")
                    print(f"      Count: {len(times)}")
        
        return self.attributions

    def single_layer(self, layer0, layer1=None):
        pass

    def make_computation_path(self, layer0, layer1=None):
        """returns path, sae0, sae1"""
        # Use base_model to access the underlying model attributes
        base_model = self.base_model
        
        if layer1 is None:
            layer1 = layer0+1
        if self.paths == PathType.BLOCK:
            assert layer0 < layer1
            assert layer0 >= 0
            assert layer1 <= base_model.gpt.config.n_layer
            if self.verbose:
                print(f'Indexing SAEs for computation path by {layer0}_act and {layer1}_act')
            sae0 = base_model.saes[f'{layer0}_act']
            sae1 = base_model.saes[f'{layer1}_act']
            
            # define function that goes from feature magnitudes in layer0 to feature magnitudes in layer1
            class Sae0Decode(nn.Module):
                def forward(self, x):
                    return sae0.decode(x)
                
            class Sae1Encode(nn.Module):
                def forward(self, x):
                    return sae1.encode(x)

            # construct function from Sae0 to Sae1
            forward_list = [Sae0Decode()] + [base_model.gpt.transformer.h[i] for i in range(layer0, layer1)] + [Sae1Encode()]
            forward = t.nn.Sequential(*forward_list)
            
            # Wrap in DataParallel if using multiple GPUs
            if len(self.device_ids) > 1 and not self.use_ddp:
                forward = DataParallel(forward, device_ids=self.device_ids)
                
            return forward, sae0, sae1

        elif self.paths == PathType.MLP:
            sae0 = base_model.saes[f'{layer0}_mlpin']
            sae1 = base_model.saes[f'{layer0}_mlpout']
            if self.verbose:
                print(f'Indexing SAEs for computation path by {layer0}_mlpin and {layer0}_mlpout')
            
            class Sae0Decode(nn.Module):
                def forward(self, x):
                    return sae0.decode(x)
                
            class Sae1Encode(nn.Module):
                def forward(self, x):
                    return sae1.encode(x)

            forward_list = [Sae0Decode(), base_model.gpt.transformer.h[layer0].mlp, Sae1Encode()]
            forward = t.nn.Sequential(*forward_list)
            
            # Wrap in DataParallel if using multiple GPUs
            if len(self.device_ids) > 1 and not self.use_ddp:
                forward = DataParallel(forward, device_ids=self.device_ids)
                
            return forward, sae0, sae1

        elif self.paths == PathType.MLP_LAYER:
            if self.verbose:
                print(f'Indexing SAEs for computation path by {layer0}_residmid and {layer0}_residpost')
            sae0 = base_model.saes[f'{layer0}_residmid']
            sae1 = base_model.saes[f'{layer0}_residpost']
            
            class Sae0Decode(nn.Module):
                def forward(self, x):
                    return sae0.decode(x)
                
            class Sae1Encode(nn.Module):
                def forward(self, x):
                    return sae1.encode(x)

            skip_list = [base_model.gpt.transformer.h[layer0].ln_2, base_model.gpt.transformer.h[layer0].mlp]
            skip = t.nn.Sequential(*skip_list)
            skip_path = SkipModule(skip)
            forward_list = [Sae0Decode(), skip_path, Sae1Encode()]
            forward = t.nn.Sequential(*forward_list)
            
            # Wrap in DataParallel if using multiple GPUs
            if len(self.device_ids) > 1 and not self.use_ddp:
                forward = DataParallel(forward, device_ids=self.device_ids)

            return forward, sae0, sae1


class IntegratedGradientAttributor(Attributor):
    def __init__(self, model: nn.Module, dataloader: TrainingDataLoader, nbatches: int = 32, 
                 steps: int = 10, verbose: bool = False, abs: bool = False, 
                 just_last: bool = False, save_dir: str = "./attributions",
                 device_ids: Optional[list] = None, use_ddp: bool = False,
                 checkpoint_every: int = 10):
        """
        Returns a dict of all consecutive integrated gradient attributions for a model.
        :param model: SparsifiedGPT model
        :param dataloader: Dataloader
        :param nbatches: How many batches of data to aggregate into attributions
        :param steps: number of steps to approximate integral with
        :param verbose: Prints updates after finishing each layer connection
        :param just_last: whether to aggregate over all sequence positions or just take last
        :param save_dir: Directory to save intermediate results
        :param device_ids: List of GPU device IDs to use
        :param use_ddp: Whether to use DistributedDataParallel
        :param checkpoint_every: Save intermediate results every N feature magnitudes
        """
        super().__init__(model, dataloader, nbatches, verbose, save_dir, device_ids, use_ddp)
        
        self.steps = steps
        self.abs = abs
        self.just_last = just_last
        self.checkpoint_every = checkpoint_every
        
        self.is_jump = (self.base_model.config.sae_variant == SAEVariant.JUMP_RELU) or \
                      (self.base_model.config.sae_variant == SAEVariant.JUMP_RELU_STAIRCASE)

    def save_checkpoint(self, attributions: Tensor, layer0: int, layer1: Optional[int], 
                       fm_start: int, fm_end: int):
        """Save intermediate checkpoint during computation"""
        start_time = time.time()
        if layer1 is None:
            checkpoint_name = f"checkpoint_layer{layer0}_fm{fm_start}-{fm_end}.pt"
        else:
            checkpoint_name = f"checkpoint_layer{layer0}-{layer1}_fm{fm_start}-{fm_end}.pt"
        checkpoint_path = os.path.join(self.save_dir, "checkpoints", checkpoint_name)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        t.save({
            'attributions': attributions,
            'fm_start': fm_start,
            'fm_end': fm_end,
            'layer0': layer0,
            'layer1': layer1
        }, checkpoint_path)
        save_time = time.time() - start_time
        if self.verbose:
            print(f"   💾 Checkpoint saved: {checkpoint_name} ({save_time:.2f}s)")

    def single_layer(self, layer0, layer1=None):
        if self.verbose:
            print(f"\n📋 Single layer computation started")
            print(f"   Source layer: {layer0}")
            print(f"   Target layer: {layer1 if layer1 is not None else layer0+1}")
        
        # Time the computation path creation
        path_start = time.time()
        forward, sae0, sae1 = self.make_computation_path(layer0, layer1)
        path_time = time.time() - path_start
        self.log_time("Computation path creation", path_time)
        
        for param in forward.parameters():
            param.requires_grad = False

        source_size, _ = sae0.W_dec.shape
        target_size, _ = sae1.W_dec.shape
        
        if self.verbose:
            print(f"   Source features: {source_size}")
            print(f"   Target features: {target_size}")
            print(f"   Total connections: {source_size * target_size:,}")
        
        # Get the primary device
        primary_device = self.device_ids[0] if self.device_ids else self.base_model.gpt.config.device
        
        attributions = t.zeros((source_size, target_size), device=primary_device)
        
        # Process in chunks for better GPU utilization
        chunk_size = max(1, target_size // (len(self.device_ids) * 4))  # Adjust chunk size based on GPU count
        
        if self.verbose:
            print(f"\n🔄 Processing {self.nbatches} batches")
            print(f"   Chunk size: {chunk_size} features")
            total_chunks = (target_size + chunk_size - 1) // chunk_size
            print(f"   Total chunks per batch: {total_chunks}")
        
        for batch_idx in range(self.nbatches):
            batch_start_time = time.time()
            
            # Data loading
            data_start = time.time()
            input, _ = self.dataloader.next_batch(primary_device)
            data_time = time.time() - data_start
            
            # Feature extraction
            feat_start = time.time()
            feature_magnitudes = get_SAE_activations(self.model, self.paths, input.long(), [layer0])
            feature_magnitudes0 = feature_magnitudes[layer0]
            feat_time = time.time() - feat_start
            
            if self.verbose and batch_idx == 0:
                print(f"\n   📊 Feature magnitudes shape: {feature_magnitudes0.shape}")
                self.log_time("Data loading", data_time)
                self.log_time("Feature extraction", feat_time)

            if self.is_jump:
                threshold = t.exp(sae0.jumprelu.log_threshold)
                base = t.where(feature_magnitudes0 > 0, threshold, 0)
            else:
                base = None

            # Process target features in chunks for better parallelization
            chunk_count = 0
            chunk_total_time = 0
            
            for chunk_start in range(0, target_size, chunk_size):
                chunk_start_time = time.time()
                chunk_end = min(chunk_start + chunk_size, target_size)
                chunk_indices = list(range(chunk_start, chunk_end))
                
                # Prepare batch of directions for parallel processing
                batch_directions = []
                for fm_i in chunk_indices:
                    y_i = t.zeros(target_size, device=primary_device)
                    y_i[fm_i] = 1
                    batch_directions.append(y_i)
                
                # Stack directions for batch processing
                batch_directions = t.stack(batch_directions)  # (chunk_size, target_size)
                
                # Process chunk in parallel
                grad_start = time.time()
                gradients = self.integrate_gradient_batch(
                    x=feature_magnitudes0,
                    fun=forward,
                    directions=batch_directions,
                    base=base
                )
                grad_time = time.time() - grad_start
                
                # Update attributions
                update_start = time.time()
                for i, fm_i in enumerate(chunk_indices):
                    if self.abs:
                        if self.just_last:
                            attributions[:, fm_i] += (gradients[i].abs()).sum(dim=0)
                        else:
                            attributions[:, fm_i] += (gradients[i].abs()).sum(dim=[0, 1])
                    else:
                        if self.just_last:
                            attributions[:, fm_i] += (gradients[i]**2).sum(dim=0)
                        else:
                            attributions[:, fm_i] += (gradients[i]**2).sum(dim=[0, 1])
                update_time = time.time() - update_start
                
                chunk_time = time.time() - chunk_start_time
                chunk_total_time += chunk_time
                chunk_count += 1
                
                # Save checkpoint if needed
                if (chunk_end % self.checkpoint_every) == 0:
                    self.save_checkpoint(attributions.clone(), layer0, layer1, 0, chunk_end)
                
                # Verbose logging for first chunk of first batch
                if self.verbose and batch_idx == 0 and chunk_count == 1:
                    print(f"\n   🧩 Chunk processing details:")
                    self.log_time("Gradient computation", grad_time)
                    self.log_time("Attribution update", update_time)
                    self.log_time("Total chunk", chunk_time)
            
            batch_time = time.time() - batch_start_time
            
            if self.verbose and (batch_idx + 1) % max(1, self.nbatches // 10) == 0:
                avg_chunk_time = chunk_total_time / chunk_count if chunk_count > 0 else 0
                print(f"\n   📈 Batch {batch_idx + 1}/{self.nbatches}")
                print(f"      Batch time: {batch_time:.2f}s")
                print(f"      Avg chunk time: {avg_chunk_time:.2f}s")
                print(f"      Estimated remaining: {(self.nbatches - batch_idx - 1) * batch_time:.1f}s")
                    
        if not self.abs:
            sqrt_start = time.time()
            attributions = t.sqrt(attributions)
            sqrt_time = time.time() - sqrt_start
            if self.verbose:
                self.log_time("Square root computation", sqrt_time)
        
        return attributions

    def integrate_gradient_batch(self, x: Tensor, fun: TensorFunction, 
                                directions: Tensor, base: Optional[Tensor] = None):
        """
        Batch version of integrate_gradient for multiple directions at once
        :param x: End of path (batchsize, seq_len, encoding)
        :param fun: Function to evaluate
        :param directions: Batch of directions (num_directions, target_size)
        :param base: Start of path
        :return: Gradients for each direction (num_directions, batch_size, seq_len, source_size)
        """
        if base is None:
            base = t.zeros_like(x)
        
        path = t.linspace(0, 1, self.steps)
        steplength = t.linalg.norm(x - base, dim=-1, keepdim=True) / self.steps
        
        batch_size, seq_len, _ = x.shape
        num_directions = directions.shape[0]
        
        # Expand directions for batch processing
        directions = directions.unsqueeze(1).unsqueeze(1).expand(num_directions, batch_size, seq_len, -1)
        if self.just_last:
            directions[:, :, :-1, :] = 0
        
        integrals = []
        for i in range(num_directions):
            integral = 0
            for alpha in path:
                point = alpha * x + (1 - alpha) * base
                point = point.detach().requires_grad_()
                
                y = fun(point)
                y.backward(retain_graph=False, gradient=directions[i])
                
                g = point.grad
                with t.no_grad():
                    if self.just_last:
                        integral += g.detach()[:, -1, :] * steplength[:, -1, :]
                    else:
                        integral += g.detach() * steplength
                        
            integrals.append(integral)
        
        return t.stack(integrals)

    def integrate_gradient(self, x: Tensor, x_i: Optional[Tensor], fun: TensorFunction, 
                          direction: Tensor, base: Optional[Tensor] = None):
        """
        Original integrate_gradient method (kept for compatibility)
        """
        if base is None:
            base = t.zeros_like(x)
        path = t.linspace(0, 1, self.steps)
        steplength = t.linalg.norm(x - base, dim=-1, keepdim=True) / self.steps

        batch_size, seq_len, _ = x.shape
        target_len = direction.shape[0]
        direction = direction.view(1, 1, target_len).expand(batch_size, seq_len, target_len)
        if self.just_last:
            direction[:, :-1, :] = 0

        integral = 0
        for alpha in path:
            point = alpha * x + (1 - alpha) * base
            point = point.detach().requires_grad_()
            y = fun(point)
            
            y.backward(retain_graph=False, gradient=direction)

            g = point.grad
            with t.no_grad():
                if x_i is None:
                    if self.just_last:
                        integral += g.detach()[:, -1, :] * steplength[:, -1, :]
                    else:
                        integral += g.detach() * steplength
                else:
                    if self.just_last:
                        integral += (g.detach()[:, -1, :] * x_i).sum(dim=-1) * steplength[:, -1, :]
                    else:
                        integral += (g.detach() * x_i).sum(dim=-1) * steplength
        return integral


class ManualAblationAttributor(Attributor):
    def __init__(self, model: nn.Module, dataloader: TrainingDataLoader, nbatches: int = 32, 
                 verbose: bool = True, epsilon: float = 0, max_size: int = 10000,
                 save_dir: str = "./attributions", device_ids: Optional[list] = None,
                 use_ddp: bool = False):
        """
        Returns a dict of all consecutive integrated gradient attributions for a model.
        :param model: SparsifiedGPT model
        :param dataloader: Dataloader
        :param nbatches: How many batches of data to aggregate into attributions
        :param verbose: Prints updates after finishing each layer connection
        :param epsilon: Threshold for feature activation
        :param max_size: Maximum size for patch lists
        :param save_dir: Directory to save intermediate results
        :param device_ids: List of GPU device IDs to use
        :param use_ddp: Whether to use DistributedDataParallel
        """
        super().__init__(model, dataloader, nbatches, verbose, save_dir, device_ids, use_ddp)
        self.epsilon = epsilon
        self.max_size = max_size

    def single_layer(self, layer0, layer1=None):
        with t.no_grad():
            if self.verbose:
                print(f"\n📋 Manual ablation computation started")
                print(f"   Source layer: {layer0}")
                print(f"   Target layer: {layer1 if layer1 is not None else layer0+1}")
            
            # Time the computation path creation
            path_start = time.time()
            forward, sae0, sae1 = self.make_computation_path(layer0, layer1)
            path_time = time.time() - path_start
            self.log_time("Computation path creation", path_time)

            source_size, _ = sae0.W_dec.shape
            target_size, _ = sae1.W_dec.shape

            if self.verbose:
                print(f"   Source features: {source_size}")
                print(f"   Target features: {target_size}")
                print(f"   Max patch size: {self.max_size}")
                print(f"   Epsilon threshold: {self.epsilon}")

            # Get the primary device
            primary_device = self.device_ids[0] if self.device_ids else self.base_model.gpt.config.device
            
            scores = t.zeros((source_size, target_size), device=primary_device)
            occurences = t.zeros((self.base_model.gpt.config.block_size, source_size), device=primary_device)

            if self.verbose:
                print(f"\n🔄 Processing {self.nbatches} batches")

            for batch_idx in range(self.nbatches):
                batch_start_time = time.time()
                
                # Data loading
                data_start = time.time()
                input, _ = self.dataloader.next_batch(primary_device)
                data_time = time.time() - data_start
                
                # Feature extraction
                feat_start = time.time()
                feature_magnitudes = get_SAE_activations(self.model, self.paths, input.long(), [layer0, layer1])
                feature_magnitudes0 = feature_magnitudes[layer0]
                feature_magnitudes1 = feature_magnitudes[layer1]
                batchsize = feature_magnitudes0.shape[0]
                feat_time = time.time() - feat_start
                
                if self.verbose and batch_idx == 0:
                    print(f"\n   📊 Feature shapes:")
                    print(f"      Source: {feature_magnitudes0.shape}")
                    print(f"      Target: {feature_magnitudes1.shape}")
                    self.log_time("Data loading", data_time)
                    self.log_time("Feature extraction", feat_time)

                # Patch creation
                patch_start = time.time()
                batch_patches = MaxSizeList(self.max_size)
                batch_indices = MaxSizeList(self.max_size)

                for b in range(batchsize):
                    up = feature_magnitudes0[b:b+1].contiguous()
                    down = feature_magnitudes1[b:b+1]
                    nz = (up > self.epsilon).nonzero(as_tuple=True)
                    for idx in zip(*nz):
                        patch = up.clone()
                        patch[0, idx[1], idx[2]] = 0
                        batch_patches.append(patch)
                        batch_indices.append((b, idx[1], idx[2]))
                
                batch_patches_list = batch_patches._get_valueless_list()
                batch_indices_list = batch_indices._get_valueless_list()
                patch_time = time.time() - patch_start
                
                if self.verbose and batch_idx == 0:
                    print(f"   🩹 Created {len(batch_patches_list)} patches")
                    self.log_time("Patch creation", patch_time)

                # Run forward passes
                if batch_patches_list:
                    forward_start = time.time()
                    
                    # Process in chunks if using multiple GPUs
                    if len(self.device_ids) > 1:
                        chunk_size = max(1, len(batch_patches_list) // len(self.device_ids))
                        all_results = []
                        
                        for i in range(0, len(batch_patches_list), chunk_size):
                            chunk = batch_patches_list[i:i+chunk_size]
                            if chunk:
                                chunk_tensor = t.cat(chunk, dim=0)
                                chunk_results = forward(chunk_tensor)
                                all_results.append(chunk_results)
                        
                        results = t.cat(all_results, dim=0) if all_results else None
                    else:
                        batch_patches_tensor = t.cat(batch_patches_list, dim=0)
                        results = forward(batch_patches_tensor)
                    
                    forward_time = time.time() - forward_start
                    
                    # Update scores
                    update_start = time.time()
                    if results is not None:
                        for i, (b, seq_idx, src_idx) in enumerate(batch_indices_list):
                            scores[src_idx] += (results[i] - feature_magnitudes1[b]).abs().sum(dim=0)
                            occurences[seq_idx, src_idx] += 1
                    update_time = time.time() - update_start
                    
                    if self.verbose and batch_idx == 0:
                        self.log_time("Forward passes", forward_time)
                        self.log_time("Score updates", update_time)

                    # Clear memory
                    del batch_patches_list
                    del batch_indices_list
                    if results is not None:
                        del results
                    t.cuda.empty_cache()
                
                batch_time = time.time() - batch_start_time
                
                # Save intermediate results periodically
                if (batch_idx + 1) % 10 == 0:
                    checkpoint_start = time.time()
                    intermediate_save_path = os.path.join(
                        self.save_dir, 
                        f"intermediate_scores_layer{layer0}-{layer1}_batch{batch_idx+1}.pt"
                    )
                    t.save({'scores': scores, 'occurences': occurences}, intermediate_save_path)
                    checkpoint_time = time.time() - checkpoint_start
                    
                    if self.verbose:
                        print(f"\n   📈 Batch {batch_idx + 1}/{self.nbatches}")
                        print(f"      Batch time: {batch_time:.2f}s")
                        print(f"      Checkpoint save time: {checkpoint_time:.2f}s")
                        print(f"      Non-zero occurences: {(occurences > 0).sum().item():,}")
                        print(f"      Estimated remaining: {(self.nbatches - batch_idx - 1) * batch_time:.1f}s")

            return scores