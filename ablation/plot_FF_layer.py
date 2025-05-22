
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
    data_dir = Path.cwd() / 'ablation' / 'data' / 'ff_layer_50P'
    method_labels = ["0.0ep00", "1.0e-03", "1.2e-03", "1.5e-03", "1.8e-03", "2.2e-03", "2.7e-03", "3.3e-03", "3.9e-03", "4.7e-03", "5.6e-03", "6.8e-03", "1.0e-02"]
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
    plot_method_labels = [["TopK","JSAE"],["TopK","JSAE"],["TopK","JSAE"],["TopK","JSAE"]]
    plot_method_indices = [[0,4],[0,2],[0,7],[0,5]] # Select the JSAE methods for each layer

    palette = sns.color_palette("colorblind", 30)
    line_styles = ['-', '--', '-.', ':', '--', ':', '-.', ':', '--', ':', '--', ':', '--', ':']
    markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', '>', 'p', 'h', '+', 'x']

    for layer_idx in range(4):
        plt.figure(figsize=(9, 6))
        # plt.figure(figsize=(12, 9))
        for idx, method_idx in enumerate(plot_method_indices[layer_idx]):
            plt.fill_between(
                feature_counts_list[method_idx],
                mean_values[layer_idx][method_idx],
                alpha=0.2,
                color=palette[idx],
                label=None
            )
            
            plt.errorbar(
                feature_counts_list[method_idx], 
                mean_values[layer_idx][method_idx],
                yerr=[yerr_lower_values[layer_idx][method_idx], yerr_upper_values[layer_idx][method_idx]],
                fmt=f'{markers[method_idx]}{line_styles[method_idx]}',
                color=palette[idx],
                capsize=3,
                markersize=4,
                label=f"{plot_method_labels[layer_idx][idx]} (AUC: {auc_values[layer_idx][method_idx]/1e3:.3g} ± {auc_errors[layer_idx][method_idx]/1e3:.2g})e3"
            )
        
        x_values = feature_counts_list[method_idx]
        num_ticks = 6
        indices = np.linspace(0, len(x_values)-1, num_ticks).astype(int)
        tick_positions = [x_values[0]] + [50000, 100000, 150000, 200000] + [x_values[-1]]
        plt.xticks(tick_positions, fontsize=18)
        plt.yticks(fontsize=18)
        
        plt.xlabel('Number of Edges Remaining', fontsize=22)
        plt.xscale('log')
        plt.ylabel('KL Divergence', fontsize=22)
        plt.title(f'SAE Pair Between FF Layer at Block {layers[layer_idx]}', fontsize=26)
        plt.legend(loc='lower left', fontsize=18)
        plt.grid(True)

        filename = Path.cwd() / 'ablation' / 'plots' / f'FF_LAYER_{layers[layer_idx]}.svg'
        plt.savefig(filename, format="svg", bbox_inches='tight')

if __name__ == "__main__":
    main()