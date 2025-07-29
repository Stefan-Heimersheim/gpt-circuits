# %%
"""
Interactive blind comparison between staircase and topk SAE features.
"""

import random
import json
import sys
import termios
import tty
import os
import re
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer
import numpy as np

# Import our visualization function
from feature_vis import load_feature_results, visualize_top_activations

def getch():
    """Get a single character from stdin without pressing enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def get_user_choice():
    """Get user preference with single keypress."""
    print("\n" + "="*60)
    print("Which set of examples do you prefer?")
    print("  1 - First set is better")
    print("  2 - Second set is better") 
    print("  3 - I am indifferent")
    print("  q - Quit")
    print("="*60)
    print("Press 1, 2, 3, or q (no enter needed): ", end="", flush=True)
    
    while True:
        choice = getch().lower()
        if choice in ['1', '2', '3', 'q']:
            print(choice)  # Echo the choice
            return choice
        # If invalid key, just continue waiting (no message to avoid clutter)

def load_comparison_data():
    """Load the feature data for both models."""
    print("Loading feature data...")
    
    interp_dict_staircase = load_feature_results('staircase.tblock.gpt2', 7, 30, min_examples=10)
    interp_dict_topk = load_feature_results('topk.tblock.gpt2', 7, 30, min_examples=10)
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokens = np.load('/workspace/gpt-circuits/data/fineweb_edu_10b/val_chunked_1024.npy')
    
    return interp_dict_staircase, interp_dict_topk, tokenizer, tokens

def get_random_feature_pair(dict_a, dict_b):
    """Get a random feature from each dictionary (with replacement)."""
    features_a = list(dict_a.keys())
    features_b = list(dict_b.keys())
    
    if not features_a or not features_b:
        return None, None
    
    feat_a = random.choice(features_a)
    feat_b = random.choice(features_b)
    
    return feat_a, feat_b

def display_side_by_side(first_feat_id, first_dict, second_feat_id, second_dict, tokens, tokenizer, k_to_show):
    """Display two feature sets side by side in columns."""
    
    # Get formatted strings for both features
    first_text = visualize_top_activations(
        first_feat_id, first_dict, tokens, tokenizer,
        k_to_show=k_to_show, show_metadata=False, show_feat_id=False, return_string=True
    )
    
    second_text = visualize_top_activations(
        second_feat_id, second_dict, tokens, tokenizer, 
        k_to_show=k_to_show, show_metadata=False, show_feat_id=False, return_string=True
    )
    
    # Split into lines
    first_lines = first_text.split('\n')
    second_lines = second_text.split('\n')
    
    # Make both lists same length by padding with empty strings
    max_lines = max(len(first_lines), len(second_lines))
    while len(first_lines) < max_lines:
        first_lines.append("")
    while len(second_lines) < max_lines:
        second_lines.append("")
    
    # Calculate column width (half screen minus separator)
    import shutil
    try:
        terminal_width = shutil.get_terminal_size().columns
        col_width = (terminal_width - 5) // 2  # -5 for separator " | "
    except:
        col_width = 60  # fallback width
    
    # Print headers
    print(f"{'🅰️  FIRST SET':<{col_width}} | {'🅱️  SECOND SET'}")
    print("=" * col_width + " | " + "=" * col_width)
    
    # Print lines side by side
    for first_line, second_line in zip(first_lines, second_lines):
        # Truncate lines that are too long and remove ANSI codes for width calculation
        first_clean = re.sub(r'\033\[[0-9;]*m', '', first_line)
        second_clean = re.sub(r'\033\[[0-9;]*m', '', second_line)
        
        if len(first_clean) > col_width:
            # Keep ANSI codes but truncate display
            excess = len(first_clean) - col_width + 3  # +3 for "..."
            first_line = first_line[:-excess] + "..."
        
        if len(second_clean) > col_width:
            excess = len(second_clean) - col_width + 3
            second_line = second_line[:-excess] + "..."
        
        # Pad first column to exact width
        first_display = first_line + " " * max(0, col_width - len(first_clean))
        print(f"{first_display} | {second_line}")

def trim_to_same_length(examples_a, examples_b):
    """Trim both example lists to the same length."""
    min_len = min(len(examples_a), len(examples_b))
    return examples_a[:min_len], examples_b[:min_len]

def save_results(results, name, filename_prefix="blind_comparison_results"):
    """Save results to disk with user name in filename."""
    # Clean the name for use in filename (remove spaces, special chars)
    clean_name = re.sub(r'[^\w\-_]', '_', name.strip().lower())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{clean_name}_{timestamp}.json"
    
    results_path = Path("data") / filename
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add user name to results
    results['user_name'] = name
    results['session_end'] = datetime.now().isoformat()
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")

def run_blind_comparison():
    """Main comparison loop."""
    print("🎯 SAE Feature Blind Comparison Study")
    print("="*50)
    
    # Get user name for the experiment
    user_name = input("Enter your name for this experiment: ").strip()
    while not user_name:
        user_name = input("Please enter a name: ").strip()
    
    print(f"Hi {user_name}! Starting your blind comparison session...")
    print("")
    print("You will see pairs of feature activations.")
    print("Your task is to choose which set you prefer.")
    print("The models are anonymized - you won't know which is which!")
    print("")
    print("💡 Instructions:")
    print("   • Just press 1, 2, or 3 (no enter needed!)")  
    print("   • 1 = First set better, 2 = Second set better, 3 = Indifferent")
    print("   • Screen clears after each vote - go as fast as you want!")
    print("   • Press 'q' to quit anytime")
    print("="*50)
    
    # Load data
    import numpy as np  # Import here since we need it
    dict_staircase, dict_topk, tokenizer, tokens = load_comparison_data()
    
    # Initialize results tracking
    results = {
        'session_start': datetime.now().isoformat(),
        'comparisons': [],
        'summary': {
            'total_comparisons': 0,
            'staircase_wins': 0,
            'topk_wins': 0,
            'indifferent': 0
        }
    }
    
    comparison_count = 0
    
    # With replacement - unlimited comparisons possible
    total_features = len(dict_staircase) + len(dict_topk)
    print(f"Features available: {len(dict_staircase)} staircase, {len(dict_topk)} topk")
    print(f"Sampling with replacement - unlimited comparisons possible!")
    print(f"\n🚀 Ready to start? Press any key...")
    getch()
    
    try:
        while True:
            comparison_count += 1
            
            # Clear screen for clean comparison view
            os.system('clear')
            
            print(f"{'='*20} COMPARISON #{comparison_count} {'='*20}")
            print(f"📊 Comparisons completed: {comparison_count-1}")
            print("="*60)
            
            # Get random features (with replacement)
            feat_staircase, feat_topk = get_random_feature_pair(
                dict_staircase, dict_topk
            )
            if feat_staircase is None or feat_topk is None:
                print("🚨 No features available!")
                break
            
            # Get examples and trim to same length
            examples_staircase = dict_staircase[feat_staircase]
            examples_topk = dict_topk[feat_topk]
            examples_staircase, examples_topk = trim_to_same_length(examples_staircase, examples_topk)
            
            # Randomly decide order (flip coin)
            show_staircase_first = random.choice([True, False])
            
            if show_staircase_first:
                first_examples = examples_staircase
                first_feat_id = feat_staircase
                first_model = 'staircase'
                second_examples = examples_topk  
                second_feat_id = feat_topk
                second_model = 'topk'
            else:
                first_examples = examples_topk
                first_feat_id = feat_topk
                first_model = 'topk'
                second_examples = examples_staircase
                second_feat_id = feat_staircase
                second_model = 'staircase'
            
            # Create temporary dictionaries for visualization
            temp_dict_first = {first_feat_id: first_examples}
            temp_dict_second = {second_feat_id: second_examples}
            
            # Display both sets side by side
            display_side_by_side(
                first_feat_id, temp_dict_first, 
                second_feat_id, temp_dict_second,
                tokens, tokenizer, len(first_examples)
            )
            
            # Get user choice
            choice = get_user_choice()
            
            if choice == 'q':
                print("\n👋 Quitting study. Thanks for your participation!")
                break
            
            # Record result (secretly)
            if choice == '1':  # First is better
                winner = first_model
            elif choice == '2':  # Second is better  
                winner = second_model
            else:  # Indifferent
                winner = 'indifferent'
            
            # Store comparison data (convert numpy types to native Python types)
            comparison_data = {
                'comparison_id': comparison_count,
                'timestamp': datetime.now().isoformat(),
                'staircase_feature': int(feat_staircase),  # Convert numpy type to int
                'topk_feature': int(feat_topk),           # Convert numpy type to int
                'staircase_shown_first': show_staircase_first,
                'user_choice': choice,
                'winner': winner,
                'num_examples': len(first_examples)
            }
            
            results['comparisons'].append(comparison_data)
            
            # Update summary
            results['summary']['total_comparisons'] = comparison_count
            if winner == 'staircase':
                results['summary']['staircase_wins'] += 1
            elif winner == 'topk':
                results['summary']['topk_wins'] += 1
            else:
                results['summary']['indifferent'] += 1
            
            print(f"\n✅ Choice recorded! Moving to next comparison...")
            print(f"📈 Total completed: {comparison_count}")
            
            # Clear screen after each vote
            os.system('clear')
    
    except KeyboardInterrupt:
        print("\n\nComparison interrupted by user.")
    
    # Save results once at the end
    save_results(results, user_name)
    
    # Final summary (still blind)
    print(f"\n{'='*50}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*50}")
    print(f"Total comparisons completed: {results['summary']['total_comparisons']}")
    print(f"Model preferences recorded (check saved file for details)")
    print(f"Indifferent responses: {results['summary']['indifferent']}")
    print("\nResults saved! The actual model identities are in the saved file.")
    print("Thank you for participating in the study! 🎉")

if __name__ == "__main__":
    run_blind_comparison() 