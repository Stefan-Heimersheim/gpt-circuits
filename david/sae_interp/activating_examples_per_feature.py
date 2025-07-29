# %%
"""
Simple SAE feature data loader.
"""

import numpy as np
from pathlib import Path

from transformers import GPT2Tokenizer
from typing import Union, List
import torch

def detokenize_batch(tokens: Union[np.ndarray, torch.Tensor], 
                    tokenizer: GPT2Tokenizer = None) -> List[str]:
    """
    Detokenize a batch of token sequences in parallel.
    
    Args:
        tokens: Array of shape (batch_size, seq_len) containing token IDs
        tokenizer: GPT2Tokenizer instance. If None, creates a new one.
    
    Returns:
        List of detokenized strings, one per batch item
    """
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Convert to numpy if torch tensor
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().numpy()
    
    # Batch decode all sequences at once
    # tokenizer.batch_decode handles the parallelization internally
    decoded_strings = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    
    return decoded_strings

# Initialize tokenizer once for reuse
_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def load_sae_data(model_name: str, layer: int, 
                  data_dir: str = '/workspace/gpt-circuits/data/interp',
                  tokens_path: str = '/workspace/gpt-circuits/data/fineweb_edu_10b/val_chunked_1024.npy'):
    """Load SAE data: returns (indices, feat_mags, tokens)"""
    
    model_dir = Path(data_dir) / model_name
    indices = np.load(model_dir / f"layer_{layer}_indices.npy")
    feat_mags = np.load(model_dir / f"layer_{layer}_feat_mags.npy")
    
    tokens = np.load(tokens_path)
    if tokens.dtype == np.uint16:
        tokens = tokens.astype(np.int32)
    
    return indices, feat_mags, tokens

# %%
layer = 7
model_name = 'topk.tblock.gpt2'
indicies, feat_mags, raw_tokens = load_sae_data(model_name, layer)
num_batches = indicies.shape[0]
tokens = raw_tokens[:num_batches]
print(f"Loaded {num_batches} batches of SAE data")
print(f"{indicies.shape=} {feat_mags.shape=} {tokens.shape=}")

# %%
import numpy as np
import time
from tqdm import tqdm
import heapq
import os
import concurrent.futures
import multiprocessing # We need Manager to create a shared queue
import threading
# --- Assume data is already defined ---
# For demonstration, we'll create it. In your code, these variables


# --- Configuration (derived from data) ---
batch_size = indicies.shape[0]
seq_len = indicies.shape[1]
top_k_features_per_token = indicies.shape[2]
k = 30

# --- Worker & Progress Monitor Functions ---

def progress_monitor(queue, total, desc):
    """
    Monitors a queue for progress updates and updates a tqdm bar.
    This runs in a separate thread in the main process.
    """
    pbar = tqdm(total=total, desc=desc)
    while True:
        try:
            # This will block until a message is received from a worker
            item = queue.get()
            if item is None:  # A 'None' value is our signal to stop
                break
            pbar.update(item)
        except (KeyboardInterrupt, SystemExit):
            break
    pbar.close()

def process_chunk_dynamic(batch_range, progress_queue=None):
    """
    Worker function. If a progress_queue is provided, it reports its progress.
    """
    start_batch, end_batch = batch_range
    local_heaps = {}

    for b_idx in range(start_batch, end_batch):
        # ... (inner loop logic is the same) ...
        indicies_batch = indicies[b_idx]
        mags_batch = feat_mags[b_idx]
        linear_indices_batch = (b_idx * seq_len) + np.arange(seq_len, dtype=np.int32)
        flat_indices = indicies_batch.flatten()
        flat_mags = mags_batch.flatten()
        flat_linear_indices = np.repeat(linear_indices_batch, top_k_features_per_token)
        for feature_id, feat_mag, linear_idx in zip(flat_indices, flat_mags, flat_linear_indices):
            example = (feat_mag, int(linear_idx))
            if feature_id not in local_heaps: local_heaps[feature_id] = [example]
            else:
                heap = local_heaps[feature_id]
                if len(heap) < k: heapq.heappush(heap, example)
                elif feat_mag > heap[0][0]: heapq.heapreplace(heap, example)
        
        # If this is the reporting worker, send a progress update.
        if progress_queue:
            progress_queue.put(1)
            
    return local_heaps

# --- Main Execution (Corrected) ---
def main_dynamic():
    print(f"\n--- Parallel Discovery with Real-Time Progress (k={k}) ---")
    start_time = time.time()

    num_workers = os.cpu_count() or 4
    chunk_size = (batch_size + num_workers - 1) // num_workers
    batch_ranges = [(i * chunk_size, min((i + 1) * chunk_size, batch_size)) for i in range(num_workers)]
    
    print(f"Splitting {batch_size} batches for {num_workers} workers.")
    
    with multiprocessing.Manager() as manager:
        progress_queue = manager.Queue()

        # ## CHANGE: Start the monitoring thread ##
        # It will watch the queue and update the bar in real-time.
        first_worker_batch_count = batch_ranges[0][1] - batch_ranges[0][0]
        monitor_thread = threading.Thread(target=progress_monitor, 
                                          args=(progress_queue, first_worker_batch_count, "Map: Worker 1 Batches"))
        monitor_thread.start()

        # --- MAP PHASE ---
        local_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_chunk = {}
            # Submit the first worker WITH the queue
            future_to_chunk[executor.submit(process_chunk_dynamic, batch_ranges[0], progress_queue)] = batch_ranges[0]
            # Submit the rest of the workers WITHOUT the queue
            for brange in batch_ranges[1:]:
                future_to_chunk[executor.submit(process_chunk_dynamic, brange)] = brange
            
            # This outer loop now just tracks completed chunks
            for future in tqdm(concurrent.futures.as_completed(future_to_chunk), total=len(batch_ranges), desc="Map: Completed Chunks"):
                try:
                    local_results.append(future.result())
                except Exception as e:
                    print(f"\nWorker process failed: {e}")

        # ## CHANGE: All workers are done. Tell the monitor thread to exit. ##
        progress_queue.put(None)  # Send the sentinel value
        monitor_thread.join()     # Wait for the monitor thread to finish cleanly

    # --- REDUCE PHASE ---
    print("\nReduce: Merging results from all workers...")
    global_heaps = {}
    for local_heaps in tqdm(local_results, desc="Merging Heaps"):
        for feat_id, heap in local_heaps.items():
            if feat_id not in global_heaps:
                global_heaps[feat_id] = heap
                heapq.heapify(global_heaps[feat_id])
            else:
                for example in heap:
                    if len(global_heaps[feat_id]) < k: heapq.heappush(global_heaps[feat_id], example)
                    elif example[0] > global_heaps[feat_id][0][0]: heapq.heapreplace(global_heaps[feat_id], example)
    
    print(f"\nFound top activations for {len(global_heaps)} unique features.")
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds.")
    return global_heaps
    # (Final result formatting would go here)

global_heaps = main_dynamic()
# %%

def convert_heaps_to_readable_format(heaps_dict, sequence_length):
    """
    Converts a dictionary of heaps from (feat_mag, linear_idx) to a sorted
    list of (feat_mag, batch_idx, seq_idx).

    Args:
        heaps_dict (dict): The dictionary where keys are feature IDs and values
                           are min-heaps of (feat_mag, linear_idx) tuples.
        sequence_length (int): The sequence length of the original data (e.g., 1024),
                               needed to reverse the linear index.

    Returns:
        dict: A new dictionary where keys are feature IDs and values are lists
              of (feat_mag, batch_idx, seq_idx) tuples, sorted by feat_mag descending.
    """
    print("\nConverting final results to (magnitude, batch, sequence) format...")
    
    final_results = {}
    
    # Use tqdm for a progress bar if the dictionary is large
    for feat_id, heap in tqdm(heaps_dict.items(), desc="Formatting Results"):
        
        converted_examples = []
        for feat_mag, linear_idx in heap:
            # Reverse the linear index calculation
            batch_idx = linear_idx // sequence_length
            seq_idx = linear_idx % sequence_length
            
            converted_examples.append((feat_mag, batch_idx, seq_idx))
        
        # Sort the final list for this feature by magnitude, descending
        converted_examples.sort(key=lambda x: x[0], reverse=True)
        
        final_results[feat_id] = converted_examples
        
    return final_results
# %%
nice_global_heaps = convert_heaps_to_readable_format(global_heaps, seq_len)

# Save the global heaps to disk for later use
import pickle
from datetime import datetime

# Create a timestamp for the filename
output_filename = f"layer_{layer}_top_{k}_features.pkl"
output_path = Path("data") / model_name / output_filename

# Ensure the directory exists
output_path.parent.mkdir(parents=True, exist_ok=True)

# Save the processed heaps
print(f"Saving global heaps to {output_path}")
with open(output_path, 'wb') as f:
    pickle.dump(nice_global_heaps, f)

print(f"Successfully saved {len(nice_global_heaps)} feature heaps to {output_path}")


# %%

# print(f"\n--- Starting Dynamic Top-K Tracking (k={k}) ---")
# print("This will dynamically build a leaderboard for every feature encountered.")
# start_time = time.time()

# # Step 1: Initialize a dictionary to hold a min-heap for each feature.
# # The dictionary will only grow to include features that are actually active.
# # Key: feature_id, Value: a min-heap list of (feat_mag, batch_idx, seq_idx) tuples.
# feature_heaps = {}

# # Step 2: Iterate through all activations in the dataset, batch by batch.
# print("Scanning dataset to find top-k activations for each feature...")
# for batch_idx in tqdm(range(indicies.shape[0]), desc="Processing Batches"):
#     for seq_idx in range(indicies.shape[1]):
#         for k_idx in range(indicies.shape[2]):
#             feature_id = indicies[batch_idx, seq_idx, k_idx]
#             feat_mag = feat_mags[batch_idx, seq_idx, k_idx]
            
#             # The data point we want to store
#             example = (feat_mag, batch_idx, seq_idx)

#             # Check if we have seen this feature before
#             if feature_id not in feature_heaps:
#                 # First time seeing this feature, create a new heap for it
#                 feature_heaps[feature_id] = [example]
#                 # heapq.heapify is not needed for a single element
#             else:
#                 heap = feature_heaps[feature_id]
#                 if len(heap) < k:
#                     # The heap is not full yet, so just add the new element
#                     heapq.heappush(heap, example)
#                 else:
#                     # The heap is full. We only add the new element if it's larger
#                     # than the smallest element currently in the heap.
#                     # heap[0] is always the smallest element in a min-heap.
#                     if feat_mag > heap[0][0]:
#                         # `heapreplace` is an efficient way to pop the smallest
#                         # element and push the new one.
#                         heapq.heapreplace(heap, example)

# end_time = time.time()
# print(f"\nDataset scan complete in {end_time - start_time:.4f} seconds.")
# print(f"Found top activations for {len(feature_heaps)} unique features.")

# # Step 3: Finalize the results.
# # The heaps are currently min-heaps (smallest first). We want to present
# # the results sorted from largest to smallest.
# print("Finalizing and sorting results...")
# final_top_activations = {}
# for feat_id, heap in feature_heaps.items():
#     # Sort the heap in descending order by magnitude (the first element of the tuple)
#     # and store it in our final results dictionary.
#     sorted_examples = sorted(heap, key=lambda x: x[0], reverse=True)
#     final_top_activations[feat_id] = sorted_examples

# # --- Verification ---
# def get_token_from_indices(batch_idx, seq_idx, tokens_array):
#     if batch_idx is None: return None
#     token_id = tokens_array[batch_idx, seq_idx]
#     token_str = _tokenizer.decode([token_id])
#     return token_str, token_id

# print("\nSample results for a few features found:")
# # Get a few feature IDs from the results to display
# features_to_show = list(final_top_activations.keys())[:5]

# for feat_id in features_to_show:
#     print(f"\nFeature {feat_id}:")
#     top_examples = final_top_activations[feat_id]
    
#     for i, (magnitude, b_idx, s_idx) in enumerate(top_examples):
#         token_str, token_id = get_token_from_indices(b_idx, s_idx, tokens)
#         print(f"  Rank {i}: Mag={magnitude:.4f}, Token='{token_str}' (at batch {b_idx}, pos {s_idx})")
        

# # Save the results to a file
# save_dir = Path(data_dir) / model_name
# save_dir.mkdir(parents=True, exist_ok=True)
# save_path = save_dir / f"layer_{layer}_top_{k}_activations.pkl"

# print(f"\nSaving top-{k} activations to {save_path}")
# import pickle
# with open(save_path, 'wb') as f:
#     pickle.dump(final_top_activations, f)
# print(f"Saved {len(final_top_activations)} features with their top-{k} activations")


# %%
