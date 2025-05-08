# %%
import matplotlib.pyplot as plt

# Re-define data
models = [
    "staircase-detach",
    "topk-x8",
    "topk-x40-tied",
    "staircase-x8",
    "staircase-untied-x8",
    "topk-x40"
]
values = [0.5706, 0.5077, 0.4533, 0.4368, 0.3163, 0.2701]
params = [335_680,  330_560, 340_800, 335_680, 991_040, 1_651_520]

# models = [
#     "topk-staircase-detach (8,16,24,32,40)",
#     "topk (8,8,8,8,8)",
#     "topk-staircase-tied-all (40,40,40,40,40)",
#     "topk-staircase-tied-chunks (8,16,24,32,40)",
#     "topk-staircase-untied (8,16,24,32,40)",
#     "topk-wide (40,40,40,40,40)"
# ]
# values = [0.5706, 0.5077, 0.4533, 0.4368, 0.3163, 0.2701]
# params = [335_680,  330_560, 340_800, 335_680, 991_040, 1_651_520]


# We compare this against a few other models as a baseline: 
# The tuple notation (x_1,x_2,x_3,x_4,x_5) indicates how much larger the hidden layer in the SAE is, as compared to the embedding size. 
# All saes were TopK saes with K=10.


# staircase-detach: Allowing each layer to only optimize a single chunk, and rely on optimization pressure from other layers to 
# provide useful features. This approach had worse performance than baseline (Appendix A).
#
# topk-x8: A standard Top-K SAE trained independently per layer, with a multiplication factor of $\times 8$ for all layers
# 
# topk-x40-tied: A standard Top-K SAE with $x40$ per layer, and encoder/decoder matrices shared between layers

# Topk-staircase-tied-chunks: The Staircase SAE described above.

# TopK-stiarcase-untied: A standard Top-K SAE with independent layers, and each subsequent layer is 8 x n_embd wider than the previous one

# TopK-wide: Same as TopK, but $x40$ for all layers.


# Define unique colors for each model
colors = ['red', 'magenta', 'green', 'cyan', 'orange', 'black']

# Plotting the bar chart
plt.figure(figsize=(10, 6))
bars = plt.barh(models, values, color='skyblue')
plt.xlabel('End-to-end cross entropy increase')
plt.ylabel('Models')
plt.title('Top-K Model Performance')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Annotate each bar with the numerical value
for bar, value in zip(bars, values):
    plt.text(
        bar.get_width() - 0.08,  # Position slightly to the right of the bar
        bar.get_y() + bar.get_height() / 2,  # Center vertically on the bar
        f'{value:.4f}',  # Format the value
        va='center',  # Align vertically
        ha='left',  # Align horizontally
        fontsize=12,  # Large font size
        color='black'  # Black text color
    )

plt.tight_layout()
plt.show()

# Plotting loss vs. parameter count
plt.figure(figsize=(8, 6))

# Scatter plot with unique colors and labeled legend
for i, (model, x, y, color) in enumerate(zip(models, params, values, colors)):
    plt.scatter(x, y, color=color, label=model, s=100)  # Larger size for visibility

    # Determine annotation position dynamically
    # Alternate text position (above/below) for consecutive points
    # Horizontal offset to place text to the right of the point
    offset_x = 2  # Points to the right
    # Vertical offset magnitude
    offset_y_magnitude = 2  # Points up or down

    if i % 2 == 0:
        # Place text above the point
        text_y_offset = offset_y_magnitude
        vertical_alignment = 'bottom'  # Text anchor is at the bottom of the text
    else:
        # Place text below the point
        text_y_offset = -offset_y_magnitude
        vertical_alignment = 'top'  # Text anchor is at the top of the text

    plt.annotate(
        f'({y:.4f}, {x})',
        (x, y),
        textcoords="offset points",
        xytext=(offset_x, text_y_offset),
        ha='left',  # Horizontal alignment of text relative to anchor
        va=vertical_alignment,
        fontsize=10
    )

plt.xlabel('Parameter Count')
plt.ylabel('End-to-end Cross Entropy Increase')
plt.xlim(0, 1.7e6)
plt.title('Loss vs. Parameter Count')
plt.legend(title="Models", loc='upper right')  # Legend placed outside for clarity
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('staircase_ce_increase.svg', dpi=300)
plt.show()

# %%