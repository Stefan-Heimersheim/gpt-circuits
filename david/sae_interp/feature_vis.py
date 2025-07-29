#  %%
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

import pickle
from pathlib import Path

def filter_features_by_min_examples(feature_dict, min_examples):
    """
    Filter features to keep only those with at least min_examples.
    
    Args:
        feature_dict (dict): Dictionary mapping feature IDs to lists of examples
        min_examples (int): Minimum number of examples required to keep a feature
    
    Returns:
        dict: Filtered dictionary with only features having >= min_examples
    """
    print(f"Filtering features to keep only those with at least {min_examples} examples...")
    original_count = len(feature_dict)
    
    filtered_dict = {
        feat_id: examples 
        for feat_id, examples in feature_dict.items() 
        if len(examples) >= min_examples
    }
    
    filtered_count = len(filtered_dict)
    print(f"Filtered from {original_count} features to {filtered_count} features")
    print(f"Removed {original_count - filtered_count} features with fewer than {min_examples} examples")
    
    return filtered_dict


def load_feature_results(model_name, layer, k=30, data_dir='data', min_examples=None):
    """
    Load the saved feature results from disk.
    
    Args:
        model_name (str): Name of the model (e.g., 'topk.tblock.gpt2')
        layer (int): Layer number
        k (int): Number of top features that were saved
        data_dir (str): Base directory where results are stored
        min_examples (int, optional): If specified, filter to keep only features 
                                    with at least this many examples
    
    Returns:
        dict: Dictionary mapping feature IDs to lists of (feat_mag, batch_idx, seq_idx)
    """
    filename = f"layer_{layer}_top_{k}_features.pkl"
    filepath = Path(data_dir) / model_name / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    print(f"Loading feature results from {filepath}")
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    
    print(f"Loaded results for {len(results)} features")
    
    if min_examples is not None:
        results = filter_features_by_min_examples(results, min_examples)
    
    return results

# %%


# Helper class for terminal colors to make output pretty
class TermColors:
    """ANSI escape codes for terminal colors."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m' # Yellow
    FAIL = '\033[91m'    # Red
    ENDC = '\033[0m'     # Resets color to default
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def visualize_top_activations(
    feat_id,
    readable_results,
    tokens,
    tokenizer,
    k_to_show=10,
    context_before=20,
    context_after=10,
    show_metadata=True,
    show_feat_id=True,
    return_string=False
):
    """
    Visualizes the top activating examples for a given feature ID in a table format.

    For each of the top 'k_to_show' examples, it fetches the token context,
    decodes it while preserving token boundaries, and prints it with the
    activating token highlighted.

    Args:
        feat_id (int): The feature index to visualize.
        readable_results (dict): The dictionary from the conversion step, mapping
                                 feature IDs to lists of (feat_mag, batch_idx, seq_idx).
        tokens (np.ndarray): The full dataset of token IDs, shape (batch_size, seq_len).
        tokenizer: An initialized Hugging Face tokenizer instance (e.g., from GPT-2).
        k_to_show (int, optional): The number of top examples to display. Defaults to 10.
        context_before (int, optional): Number of tokens to show before the target. Defaults to 20.
        context_after (int, optional): Number of tokens to show after the target. Defaults to 10.
        show_metadata (bool, optional): Whether to show feat_mag, batch_idx, seq_idx columns. Defaults to True.
        show_feat_id (bool, optional): Whether to show the feature ID in the header. Defaults to True.
        return_string (bool, optional): If True, return formatted string instead of printing. Defaults to False.
    """
    if feat_id not in readable_results:
        error_msg = f"{TermColors.FAIL}Feature not found in the results dictionary.{TermColors.ENDC}"
        if return_string:
            return error_msg
        print(error_msg)
        return

    # Get the top examples for the specified feature
    top_examples = readable_results.get(feat_id, [])
    if not top_examples:
        warning_msg = f"{TermColors.WARNING}Feature has no activating examples in the results.{TermColors.ENDC}"
        if return_string:
            return warning_msg
        print(warning_msg)
        return

    # Build output lines
    output_lines = []

    # Header - conditionally show feature ID
    if show_feat_id:
        output_lines.append(f"{TermColors.HEADER}{TermColors.BOLD}--- Top {min(k_to_show, len(top_examples))} Activating Examples for Feature {feat_id} ---{TermColors.ENDC}")
    else:
        output_lines.append(f"{TermColors.HEADER}{TermColors.BOLD}--- Top {min(k_to_show, len(top_examples))} Activating Examples ---{TermColors.ENDC}")

    # Prepare table data
    table_data = []
    
    # Iterate through the top examples and collect data
    for rank, (magnitude, batch_idx, seq_idx) in enumerate(top_examples[:k_to_show], 1):
        
        # --- 1. Fetch the token context ---
        token_sequence = tokens[batch_idx]
        
        # Define the window boundaries, ensuring they don't go out of bounds
        start_idx = max(0, seq_idx - context_before)
        end_idx = min(len(token_sequence), seq_idx + context_after + 1)
        
        window_tokens = token_sequence[start_idx:end_idx]
        
        # The index of our target token WITHIN the window
        target_token_window_idx = seq_idx - start_idx

        # --- 2. Decode tokens individually to preserve boundaries ---
        decoded_tokens = [tokenizer.decode(t) for t in window_tokens]

        # --- 3. Reconstruct the text with highlighting ---
        text_before_target = "".join(decoded_tokens[:target_token_window_idx])
        target_token_text = decoded_tokens[target_token_window_idx]
        text_after_target = "".join(decoded_tokens[target_token_window_idx+1:])

        # Create highlighted version with colored background
        highlighted_text = (
            text_before_target +
            f"\033[43m\033[30m{target_token_text}\033[0m" +  # Yellow background, black text
            text_after_target
        )
        
        # Clean up text for display (remove excessive whitespace, newlines)
        display_text = highlighted_text.replace('\n', '\\n').replace('\t', '\\t')
        
        table_data.append({
            'rank': rank,
            'text': display_text,
            'feat_mag': magnitude,
            'batch_idx': batch_idx,
            'seq_idx': seq_idx
        })

    # --- 4. Build the table ---
    # Calculate column widths based on what we're showing
    text_width = min(80, max(20, max(len(row['text']) for row in table_data) + 5))
    
    # Add empty line before table
    output_lines.append("")
    
    # Print table header
    output_lines.append(f"{TermColors.BOLD}")
    if show_metadata:
        output_lines.append(f"{'Rank':<6} {'Text':<{text_width}} {'Feat Mag':<10} {'Batch':<8} {'Seq':<6}")
        output_lines.append("-" * (6 + text_width + 10 + 8 + 6 + 4))  # +4 for spacing
    else:
        output_lines.append(f"{'Rank':<6} {'Text':<{text_width}}")
        output_lines.append("-" * (6 + text_width + 2))  # +2 for spacing
    output_lines.append(f"{TermColors.ENDC}")
    
    # Build table rows
    for row in table_data:
        # Truncate text if too long
        display_text = row['text']
        if len(display_text) > text_width:
            # Find a good truncation point that preserves the highlighting
            if '\033[43m' in display_text and '\033[0m' in display_text:
                # Keep the highlighted portion
                highlight_start = display_text.find('\033[43m')
                highlight_end = display_text.find('\033[0m') + 4
                if highlight_end - highlight_start < text_width - 10:
                    # Can fit the highlight plus some context
                    context_space = text_width - (highlight_end - highlight_start) - 3  # -3 for "..."
                    before_space = context_space // 2
                    after_space = context_space - before_space
                    
                    before_text = display_text[:highlight_start][-before_space:] if highlight_start > before_space else display_text[:highlight_start]
                    after_text = display_text[highlight_end:][:after_space] if len(display_text[highlight_end:]) > after_space else display_text[highlight_end:]
                    
                    display_text = before_text + display_text[highlight_start:highlight_end] + after_text
                    if len(display_text) > text_width:
                        display_text = display_text[:text_width-3] + "..."
                else:
                    display_text = display_text[:text_width-3] + "..."
            else:
                display_text = display_text[:text_width-3] + "..."
        
        if show_metadata:
            output_lines.append(f"{row['rank']:<6} {display_text:<{text_width}} {row['feat_mag']:<10.4f} {row['batch_idx']:<8} {row['seq_idx']:<6}")
        else:
            output_lines.append(f"{row['rank']:<6} {display_text:<{text_width}}")

    if return_string:
        return "\n".join(output_lines)
    else:
        for line in output_lines:
            print(line)


if __name__ == "__main__":
    interp_dict_staircase = load_feature_results('staircase.tblock.gpt2', 7, 30, min_examples=10)
    interp_dict_topk = load_feature_results('topk.tblock.gpt2', 7, 30, min_examples=10)

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokens = np.load('/workspace/gpt-circuits/data/fineweb_edu_10b/val_chunked_1024.npy')
    visualize_top_activations(157719, interp_dict_staircase, tokens, tokenizer)
# %%

