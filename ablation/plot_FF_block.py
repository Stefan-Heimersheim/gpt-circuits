
from pathlib import Path
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Path setup
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils import load_experiments_and_extract_data, prepare_plot_data, compute_auc_values


def main():
    
    # DATA LOADING
    exp_output = 'kl_divergence_full_model'
    data_dir = Path.cwd() / 'ablation' / 'data' / 'residmid-residpost_50P_v1'
    method_labels = ["0.0ep00", "staircase"]
    edge_sort = "gradient"
    layers = [0,1,2,3]

    kl_values_list = []
    feature_counts_list = []

    for method in method_labels:
        # Load experiments and extract logits
        layer_logits_values, feature_counts = load_experiments_and_extract_data(
            exp_output,
            data_dir, 
            method,
            edge_sort,
            layers
        )
        
        # Store results
        kl_values_list.append(layer_logits_values)
        feature_counts_list.append(feature_counts)

    #  DATA PROCESSING
    # Compute KL divergence values
    mean_values, yerr_lower_values, yerr_upper_values = prepare_plot_data(
        kl_values_list,
        feature_counts_list,
        method_labels,
        layers
    )

    # Compute area under the curve for each layer and method
    auc_values, auc_errors = compute_auc_values(
        feature_counts_list,
        mean_values,
        yerr_lower_values,
        yerr_upper_values,
        method_labels,
        layers
    )

    # For each layer, print the AUC values and their error terms for each method.
    for l, layer in enumerate(layers):
        print(f"Layer {layer} AUC values:")
        for m, method in enumerate(method_labels):
            print(f"  {method}: AUC = ({auc_values[l][m]/1e5:.6f} ± {auc_errors[l][m]/1e5:.6f})e5")
        print()
    
    # PLOT RESULTS
    plot_method_labels = ["TopK", "Staircase"] # Adjusted labels for plotting

    # Get colorblind friendly palette
    palette = sns.color_palette("colorblind", len(plot_method_labels)+1)

    # Define line styles for each method
    line_styles = ['-', '--', '-.', ':', '--', ':', '-.', ':', '--', ':', '--', ':', '--', ':']

    # Define markers for each method
    markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', '>', 'p', 'h', '+', 'x']

    # Plot results for each layer
    for layer_idx in range(len(layers)):
        plt.figure(figsize=(9, 6))
        for method_idx in range(len(plot_method_labels)):
            # Add filled area under the curve
            plt.fill_between(
                feature_counts_list[method_idx],
                mean_values[layer_idx][method_idx],
                alpha=0.2,  # Transparency level
                color=palette[method_idx if method_idx==0 else 2],  # Use colorblind friendly color
                label=None  # Don't add to legend
            )
            
            # Plot the line with error bars
            plt.errorbar(
                feature_counts_list[method_idx], 
                mean_values[layer_idx][method_idx],
                yerr=[yerr_lower_values[layer_idx][method_idx], yerr_upper_values[layer_idx][method_idx]],
                fmt=f'{markers[method_idx]}{line_styles[method_idx]}',  # Use marker and line style
                color=palette[method_idx if method_idx==0 else 2],  # Use colorblind friendly color
                capsize=3,
                markersize=4,
                label=f"{plot_method_labels[method_idx]} (AUC: {auc_values[layer_idx][method_idx]/1e5:.3g} ± {auc_errors[layer_idx][method_idx]/1e5:.2g})e5"
            )
        
        # Set a subset of x-ticks for better readability
        x_values = feature_counts_list[0]  # Use first method as reference
        num_ticks = 6  # Choose reasonable number of ticks
        indices = np.linspace(0, len(x_values)-1, num_ticks).astype(int)
        # tick_positions = [x_values[i] for i in indices]
        tick_positions = [x_values[0]] + [50000, 100000, 150000, 200000] + [x_values[-1]]  # Include first and last values
        
        plt.xticks(tick_positions, fontsize=18)  # Set x-ticks
        plt.yticks(fontsize=18)  # Set y-ticks
        
        plt.xlabel('Number of Edges Remaining', fontsize=22)
        plt.xscale('log')  # Use log scale for x-axis
        plt.ylabel('KL Divergence', fontsize=22)
        plt.title(f'SAE Pair Between Transformer Block {layers[layer_idx]}', fontsize=24)
        plt.legend(loc='lower left', fontsize=18)
        plt.grid(True)

        filename = Path.cwd() / 'ablation' / 'plots' / f'FF_BLOCK_{layers[layer_idx]}_test.svg'
        plt.savefig(filename, format="svg", bbox_inches='tight')

if __name__ == "__main__":
    main()