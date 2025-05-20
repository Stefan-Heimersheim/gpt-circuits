# %%
import argparse
import glob
import re
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import itertools

from pathlib import Path

# --- Extraction Functions ---

def extract_metrics_from_eval_log(eval_log_path, num_pairs: int):
    """
    Extracts metrics from the last relevant line in an eval.log file.
    Looks for 'ce_loss_increases', '∇_l1', and 'compound_ce_loss_increase'.

    Args:
        eval_log_path: Path object pointing to the eval.log file.
        num_pairs: Expected number of pairs (for ce_loss_increases and ∇_l1).

    Returns:
        A dictionary containing the found metrics:
        {
            'ce_increases': list of num_pairs floats or None,
            'nabla_l1_values': list of num_pairs floats or None,
            'compound_ce': float or None
        }
        Returns None for a metric if it's not found or parsing fails.
    """
    results = {'ce_increases': None, 'nabla_l1_values': None, 'compound_ce': None}
    lines = []
    expected_ce_count = num_pairs
    expected_nabla_count = num_pairs

    try:
        with open(eval_log_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                print(f"Warning: Eval log file is empty: {eval_log_path}", file=sys.stderr)
                return results # Return dict with all Nones

        # Search backwards for the last line containing *any* of the metrics
        last_eval_line = None
        # Look for lines containing 'eval' and at least one keyword to be more robust
        keywords = ["ce_loss_increases", "∇_l1", "compound_ce_loss_increase"]
        for line in reversed(lines):
            # A basic check to ensure it's likely an evaluation line
            if "eval" in line and any(keyword in line for keyword in keywords):
                last_eval_line = line
                break

        if not last_eval_line:
            print(f"Warning: Could not find a relevant eval line containing any expected metrics ({', '.join(keywords)}) in {eval_log_path}", file=sys.stderr)
            return results # Return dict with all Nones

        # Refined regex to capture numbers until the next pipe or end of line
        # Handles potential extra spaces and scientific notation.
        num_pattern = r"[\-\+]?[\d\.eE]+"
        list_pattern = rf"((?:{num_pattern}\s*)+?)" # Non-greedy list of numbers

        # --- Extract ce_loss_increases (if present) ---
        # Match 'ce_loss_increases', capture numbers, stop at next '|' or end
        ce_match = re.search(rf"ce_loss_increases\s+{list_pattern}\s*(?:\||$)", last_eval_line)
        if ce_match:
            try:
                ce_str = ce_match.group(1).strip()
                # Split captured string by whitespace and convert to float
                parsed_ces = [float(v) for v in re.split(r'\s+', ce_str) if v]
                if len(parsed_ces) == expected_ce_count:
                    results['ce_increases'] = parsed_ces
                else:
                     print(f"Warning: Expected {expected_ce_count} CE increases, found {len(parsed_ces)} in {eval_log_path}. Line: '{last_eval_line.strip()}'", file=sys.stderr)
            except ValueError as e:
                 print(f"Warning: Could not parse CE increases from line in {eval_log_path}. Captured: '{ce_match.group(1)}'. Error: {e}", file=sys.stderr)
            except Exception as e:
                 print(f"Warning: Unexpected error parsing CE increases from line in {eval_log_path}. Line: '{last_eval_line.strip()}'. Error: {e}", file=sys.stderr)

        # --- Extract ∇_l1 values (if present) ---
        # Match '∇_l1', capture numbers, stop at next '|' or end
        nabla_match = re.search(rf"∇_l1\s+{list_pattern}\s*(?:\||$)", last_eval_line)
        if nabla_match:
            try:
                nabla_l1_values_str = nabla_match.group(1).strip()
                # Split captured string by whitespace and convert to float
                parsed_nablas = [float(v) for v in re.split(r'\s+', nabla_l1_values_str) if v]
                if len(parsed_nablas) == expected_nabla_count:
                    results['nabla_l1_values'] = parsed_nablas
                else:
                    print(f"Warning: Expected {expected_nabla_count} ∇_l1 values, found {len(parsed_nablas)} in {eval_log_path}. Line: '{last_eval_line.strip()}'", file=sys.stderr)
            except ValueError as e:
                print(f"Warning: Could not parse ∇_l1 values from line in {eval_log_path}. Captured: '{nabla_match.group(1)}'. Error: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Unexpected error parsing ∇_l1 values from line in {eval_log_path}. Line: '{last_eval_line.strip()}'. Error: {e}", file=sys.stderr)

        # --- Extract compound_ce_loss_increase (if present) ---
        # Match 'compound_ce_loss_increase', capture one number, stop at next '|' or end
        compound_ce_match = re.search(rf"compound_ce_loss_increase\s+({num_pattern})\s*(?:\||$)", last_eval_line)
        if compound_ce_match:
            try:
                compound_ce_str = compound_ce_match.group(1)
                results['compound_ce'] = float(compound_ce_str)
            except ValueError as e:
                print(f"Warning: Could not parse compound_ce_loss_increase from line in {eval_log_path}. String: '{compound_ce_str}'. Error: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Unexpected error parsing compound_ce_loss_increase from line in {eval_log_path}. Line: '{last_eval_line.strip()}'. Error: {e}", file=sys.stderr)

        return results

    except FileNotFoundError:
         # Don't print a warning here, handled in main loop
        return results # Return dict with all Nones
    except Exception as e:
        print(f"Error reading or processing {eval_log_path}: {e}", file=sys.stderr)
        return results # Return dict with all Nones


# --- Main Script Logic ---

def main():
    parser = argparse.ArgumentParser(
        description="Extracts CE loss increases and ∇_l1 values from eval.log files "
                    "found via a glob pattern. Generates plots showing "
                    "the i-th CE loss increase vs. the i-th ∇_l1 value, "
                    "annotated with the sparsity coefficient from the directory name."
    )
    parser.add_argument(
        "path_pattern",
        help="Glob pattern for checkpoint directories (e.g., 'checkpoints/jblock_sparse*'). Quote the pattern."
    )
    parser.add_argument(
        "--num-pairs", type=int, default=4,
        help="Expected number of pairs (default: 4). Determines how many CE/∇_l1 values to extract per log file."
    )
    parser.add_argument(
        "--log-scale-x", action="store_true",
        help="Use a logarithmic scale for the x-axis (∇_l1 Value)."
    )
    parser.add_argument(
        "--plot-mode", choices=['combined', 'separate'], default='combined',
        help="Plotting mode: 'combined' overlays all pairs on one plot (default), "
             "'separate' creates an individual plot file for each pair."
    )
    parser.add_argument(
        "--hide-plots", action="store_true",
        help="Save plots but do not display them with plt.show()."
    )
    args = parser.parse_args()
    NUM_PAIRS = args.num_pairs

    # Data structure: {pair_idx: [(nabla_l1, ce_increase, sparsity_coeff_from_dir), ...]}
    pair_data = defaultdict(list)

    potential_paths = glob.glob(args.path_pattern)
    directories = [Path(p) for p in potential_paths if Path(p).is_dir()]

    if not directories:
        print(f"Error: No directories found matching pattern '{args.path_pattern}'", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(directories)} matching directories.")
    print(f"Expecting {NUM_PAIRS} pairs of (CE Increase, ∇_l1) per directory.")

    processed_count = 0
    success_count = 0
    skipped_log_scale_points = 0
    for dir_path in sorted(directories):
        processed_count += 1
        eval_log_file = dir_path / "eval.log"
        print(f"Processing directory: {dir_path}")

        # Extract sparsity coefficient from directory name (used for annotation)
        # Example: checkpoints/jblock_sparse-1.00e-05_... -> 1.00e-05
        sparsity_match = re.search(r"sparse-([\d\.]+e[-+]?\d+)", str(dir_path))
        if not sparsity_match:
            print(f"  Warning: Could not extract sparsity coefficient (e.g., 'sparse-1.0e-5') from directory name: {dir_path}. Skipping directory.", file=sys.stderr)
            continue
        sparsity_coeff_from_dir = float(sparsity_match.group(1))

        eval_metrics = None
        valid_data_found_for_dir = False

        # --- Extract Eval Metrics ---
        if eval_log_file.is_file():
            eval_metrics = extract_metrics_from_eval_log(eval_log_file, NUM_PAIRS)
            if eval_metrics is None or all(v is None for v in eval_metrics.values()):
                 print(f"  Info: Could not get any valid metrics from {eval_log_file}.")
        else:
            print(f"  Warning: eval.log not found in {dir_path}")

        # --- Store Data for Plotting ---
        if (eval_metrics is not None and
            eval_metrics.get('ce_increases') is not None and
            eval_metrics.get('nabla_l1_values') is not None):

            ce_increases = eval_metrics['ce_increases']
            nabla_l1_values = eval_metrics['nabla_l1_values']

            print(f"  Successfully extracted data for {NUM_PAIRS} pairs.")
            success_count += 1
            valid_data_found_for_dir = True
            points_added_for_dir = 0
            for i in range(NUM_PAIRS):
                # Check individual nabla value for log scale compatibility if requested
                if args.log_scale_x and nabla_l1_values[i] <= 0:
                     print(f"  Info: Skipping pair {i+1} data point from {dir_path} due to non-positive ∇_l1 value ({nabla_l1_values[i]}) required for log scale.", file=sys.stderr)
                     skipped_log_scale_points += 1
                     continue # Skip only this point for this pair

                ce_increase = ce_increases[i]
                nabla_l1 = nabla_l1_values[i]
                # Store (nabla_l1, ce_increase, sparsity_coeff_from_dir)
                pair_data[i].append((nabla_l1, ce_increase, sparsity_coeff_from_dir))
                points_added_for_dir += 1
            if points_added_for_dir == 0 and valid_data_found_for_dir:
                # If we extracted data but skipped all points (e.g., due to log scale)
                success_count -= 1 # Decrement success count as no points were actually stored
                print(f"  Warning: All points from {dir_path} were skipped (e.g., due to log scale requirements).")


        elif eval_metrics is not None: # Failed extraction, provide reason
             reason = "unknown issue"
             if eval_metrics.get('ce_increases') is None and eval_metrics.get('nabla_l1_values') is None:
                 reason = "missing both ce_loss_increases and ∇_l1 values"
             elif eval_metrics.get('ce_increases') is None:
                 reason = "missing ce_loss_increases"
             elif eval_metrics.get('nabla_l1_values') is None:
                 reason = "missing ∇_l1 values"
             print(f"  Failed to extract complete data for plotting ({reason}) for {dir_path}.")
        # else: Handled by eval_metrics is None check or file not found warning

    print(f"\nProcessed {processed_count} directories. Successfully extracted and stored data from {success_count} directories.")
    if skipped_log_scale_points > 0:
        print(f"Skipped {skipped_log_scale_points} individual data points due to non-positive ∇_l1 values required for log scale.")

    # --- Sanitize Filename Pattern ---
    base_filename_pattern = args.path_pattern.replace('*', '_star_').replace('?', '_qmark_')
    base_filename_pattern = re.sub(r'[\\/:\s]', '_', base_filename_pattern)
    base_filename_pattern = re.sub(r'[^a-zA-Z0-9_\-\.]', '', base_filename_pattern)
    base_filename_pattern = base_filename_pattern.strip('_.-')

    plot_files = []

    # --- Check if any data was collected ---
    if not pair_data or all(not points for points in pair_data.values()):
        print("\nError: No valid data points could be extracted or stored for any pair.", file=sys.stderr)
        sys.exit(1)

    # --- Plotting ---
    if args.plot_mode == 'combined':
        print("\nGenerating combined plot for all pairs...")
        plt.figure(figsize=(12, 8))
        any_pair_plotted = False
        all_nablas_combined = []
        # Define markers and colors for different pairs
        markers = itertools.cycle(['o', 's', '^', 'v', 'D', 'P', '*', 'X'])
        # Use a colormap for colors (e.g., tab10 or viridis)
        colors = plt.cm.tab10(np.linspace(0, 1, NUM_PAIRS)) # Use tab10 for distinct colors

        for i in range(NUM_PAIRS):
            points = pair_data[i] # List of (nabla_l1, ce_increase, sparsity_coeff_from_dir)
            if not points:
                print(f"  No data points found or stored for Pair {i+1}.")
                continue

            any_pair_plotted = True
            print(f"  Plotting {len(points)} points for Pair {i+1}...")

            # Sort points by nabla_l1 (x-axis) for consistent plotting lines
            points.sort(key=lambda x: x[0])

            # Extract data after sorting
            nablas = [p[0] for p in points] # x-axis (∇_l1 values)
            ce_increases = [p[1] for p in points]  # y-axis
            coeffs_from_dir = [p[2] for p in points] # sparsity coefficients for labels
            all_nablas_combined.extend(nablas) # Collect all x-values for log scale check

            marker = next(markers)
            color = colors[i]

            # Plot points for this pair
            plt.scatter(nablas, ce_increases, zorder=5, label=f"Pair {i+1}", marker=marker, color=color)
            # Plot connecting line for this pair
            plt.plot(nablas, ce_increases, linestyle='--', marker=marker, zorder=4, alpha=0.7, label='_nolegend_', color=color)

            # Label points with sparsity coefficient from directory name
            for j, coeff in enumerate(coeffs_from_dir):
                # Adjust text position slightly to avoid overlap
                plt.text(nablas[j], ce_increases[j] * 1.01, f' {coeff:.1e}', fontsize=7, rotation=0, ha='left', va='bottom', color=color)

        if not any_pair_plotted:
            print("\nError: No valid data points found for any pair to plot.", file=sys.stderr)
            plt.close() # Close the empty figure
            sys.exit(1)

        xlabel = "∇_l1 Value"
        if args.log_scale_x:
            # Check if *all* plotted points are positive
            valid_log_data = [n for n in all_nablas_combined if n > 0]
            if len(valid_log_data) == len(all_nablas_combined):
                try:
                    plt.xscale('log')
                    xlabel += " (Log Scale)"
                    print("  Using log scale for x-axis.")
                except ValueError as e:
                     print(f"  Warning: Cannot set log scale for combined plot x-axis: {e}. Using linear scale.", file=sys.stderr)
            elif valid_log_data:
                print(f"  Warning: Cannot set log scale for combined plot x-axis because {len(all_nablas_combined) - len(valid_log_data)} non-positive value(s) were plotted. Using linear scale.", file=sys.stderr)
            else: # Should not happen if skip logic worked, but good failsafe
                print(f"  Warning: Cannot set log scale for combined plot x-axis as all plotted values are non-positive. Using linear scale.", file=sys.stderr)

        plt.xlabel(xlabel)
        plt.ylabel("Cross Entropy Increase")
        plt.title(f"Combined Pairs: CE Increase vs. ∇_l1 Value\n(Point labels show sparsity coefficient from directory name)")
        plt.grid(True, which="both", linestyle='--', alpha=0.6)
        plt.legend(fontsize='small', title="Pairs")
        plt.tight_layout()

        save_path = f"{base_filename_pattern}_combined_pairs_pareto.png"
        plot_files.append(save_path)
        print(f"\nSaving combined plot to: {save_path}")
        try:
            plt.savefig(save_path)
            print("Plot saved successfully.")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}", file=sys.stderr)

        plt.close()

    elif args.plot_mode == 'separate':
        print("\nGenerating separate plot for each pair...")
        any_pair_plotted = False
        for i in range(NUM_PAIRS):
            points = pair_data[i] # List of (nabla_l1, ce_increase, sparsity_coeff_from_dir)
            if not points:
                print(f"  No data points found or stored for Pair {i+1}. Skipping plot.")
                continue

            any_pair_plotted = True
            print(f"  Generating plot for Pair {i+1} ({len(points)} points)...")

            plt.figure(figsize=(10, 6)) # Create a new figure for each pair

            # Sort points by nabla_l1 (x-axis) for consistent plotting lines
            points.sort(key=lambda x: x[0])

            # Extract data after sorting
            nablas = [p[0] for p in points] # x-axis (∇_l1 values)
            ce_increases = [p[1] for p in points]  # y-axis
            coeffs_from_dir = [p[2] for p in points] # sparsity coefficients for labels

            # Plot points for this pair
            plt.scatter(nablas, ce_increases, zorder=5, marker='o', color='blue')
            # Plot connecting line for this pair
            plt.plot(nablas, ce_increases, linestyle='--', marker='o', zorder=4, alpha=0.7, color='blue')

            # Label points with sparsity coefficient from directory name
            for j, coeff in enumerate(coeffs_from_dir):
                # Adjust text position slightly to avoid overlap
                plt.text(nablas[j], ce_increases[j] * 1.01, f' {coeff:.1e}', fontsize=8, rotation=0, ha='left', va='bottom', color='black')

            xlabel = "∇_l1 Value"
            if args.log_scale_x:
                # Check if *all* points for *this pair* are positive
                valid_log_data = [n for n in nablas if n > 0]
                if len(valid_log_data) == len(nablas):
                    try:
                        plt.xscale('log')
                        xlabel += " (Log Scale)"
                        print(f"    Using log scale for x-axis for Pair {i+1}.")
                    except ValueError as e:
                         print(f"    Warning: Cannot set log scale for Pair {i+1} x-axis: {e}. Using linear scale.", file=sys.stderr)
                elif valid_log_data:
                    print(f"    Warning: Cannot set log scale for Pair {i+1} x-axis because {len(nablas) - len(valid_log_data)} non-positive value(s) were plotted. Using linear scale.", file=sys.stderr)
                else:
                    print(f"    Warning: Cannot set log scale for Pair {i+1} x-axis as all plotted values are non-positive. Using linear scale.", file=sys.stderr)

            plt.xlabel(xlabel)
            plt.ylabel("Cross Entropy Increase")
            plt.title(f"Pair {i+1}: CE Increase vs. ∇_l1 Value\n(Point labels show sparsity coefficient from directory name)")
            plt.grid(True, which="both", linestyle='--', alpha=0.6)
            plt.tight_layout()

            save_path = f"{base_filename_pattern}_pair_{i+1}_pareto.png"
            plot_files.append(save_path)
            print(f"  Saving plot for Pair {i+1} to: {save_path}")
            try:
                plt.savefig(save_path)
                print("  Plot saved successfully.")
            except Exception as e:
                print(f"  Error saving plot to {save_path}: {e}", file=sys.stderr)

            plt.close() # Close the figure for this pair

        if not any_pair_plotted:
             print("\nError: No valid data points found for any pair to plot.", file=sys.stderr)
             sys.exit(1)


    # --- Final Message ---
    if plot_files:
        print(f"\n{len(plot_files)} plot(s) saved:")
        for f in plot_files:
            print(f"  - {f}")
        if not args.hide_plots:
            print("\nDisplaying plots is disabled by default when saving files.")
            # plt.show() # Uncomment if you want to force display even when saving
        else:
             print("\nDisplaying plots was disabled by --hide-plots.")
    elif not plot_files: # Should only happen if plotting failed or no data
         print("\nNo plots were generated or saved.")

if __name__ == "__main__":
    main()
# %%