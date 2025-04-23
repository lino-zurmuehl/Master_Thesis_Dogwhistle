import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Load your CSV
df_gemma = pd.read_csv("eval/gemma_original_predictions.csv")  
df_roberta = pd.read_csv("eval/roberta_original_predictions.csv")
# Ensure correct types
df_gemma['true_label'] = df_gemma['true_label'].astype(int)
df_gemma['probability'] = df_gemma['probability'].astype(float)

df_roberta['true_label'] = df_roberta['true_label'].astype(int)
df_roberta['probability'] = df_roberta['probability'].astype(float)
# ROC curve Gemma
fpr_gemma, tpr_gemma, _ = roc_curve(df_gemma['true_label'], df_gemma['probability'])
roc_auc_gemma = auc(fpr_gemma, tpr_gemma)

# ROC curve for RoBERTa
fpr_roberta, tpr_roberta, _ = roc_curve(df_roberta['true_label'], df_roberta['probability'])
roc_auc_roberta = auc(fpr_roberta, tpr_roberta)

sns.set(style="whitegrid", font="Courier New")
plt.figure(figsize=(6, 5))
plt.plot(fpr_gemma, tpr_gemma, color='#fc8d62', label=f'Gemma (AUC = {roc_auc_gemma:.2f})')
plt.plot(fpr_roberta, tpr_roberta, color='#66c2a5', label=f'RoBERTa (AUC = {roc_auc_roberta:.2f})')
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('plots/eval_roc_curve_combined.png', dpi=300, bbox_inches='tight')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib.colors import ListedColormap

# --- Basic Setup ---

# --- Define Colors ---
# Using Set2 palette: set2[1] is greenish, set2[0] is reddish
set2 = sns.color_palette("Set2")
incorrect_color = set2[0] # Reddish (for FP, FN)
correct_color = set2[1]   # Greenish (for TN, TP)

# Create a colormap: Index 0 maps to incorrect_color, Index 1 maps to correct_color
# This map will be used with the 'correctness_matrix' later
custom_cmap = ListedColormap([incorrect_color, correct_color])

# --- Load and Prepare Data ---
try:
    roberta_df = pd.read_csv("eval/roberta_external_predictions.csv")
    gemma_df = pd.read_csv("eval/gemma_external_predictions.csv")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure 'eval/roberta_external_predictions.csv' and 'eval/gemma_external_predictions.csv' exist.")
    exit() # Exit if files are missing

# Convert label columns to integers (can be done in one go)
for df in [roberta_df, gemma_df]:
    df['true_label'] = df['true_label'].astype(int)
    df['predicted_label'] = df['predicted_label'].astype(int)

# --- Compute Confusion Matrices ---
cm_roberta = confusion_matrix(roberta_df['true_label'], roberta_df['predicted_label'])
cm_gemma = confusion_matrix(gemma_df['true_label'], gemma_df['predicted_label'])
labels = ["Negative (0)", "Positive (1)"] # Assuming 0=Negative, 1=Positive

# --- Simplified Plotting Function ---
def plot_simplified_confmat(cm, ax, title, show_yticklabels, cmap):
    """Plots a confusion matrix with specific colors for correct/incorrect predictions."""
    if cm.shape != (2, 2):
        raise ValueError("This function is designed for 2x2 confusion matrices.")

    # Create a matrix indicating correctness (1 for diagonal/correct, 0 for off-diagonal/incorrect)
    # This matrix determines the background color via the cmap.
    correctness_matrix = np.array([[1, 0], [0, 1]]) # TN=1, FP=0, FN=0, TP=1

    sns.heatmap(
        correctness_matrix, # Use this simple matrix for cell *coloring*
        annot=cm,           # Use the *actual* confusion matrix for numbers
        fmt="d",            # Format numbers as integers
        cmap=cmap,          # Use our Red/Green map ([incorrect, correct])
        cbar=False,         # No color bar needed for binary correctness
        xticklabels=labels,
        yticklabels=labels if show_yticklabels else False,
        linewidths=1,
        linecolor='gray',
        ax=ax,
        square=True,        # Ensure cells are square
        annot_kws={"size": 14, "color": "black"}, # Adjust annotation appearance
        vmin=0,             # Explicitly set scale for the correctness_matrix
        vmax=1
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    if show_yticklabels:
        ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=10)

# --- Create and Save Plot ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5)) # Adjusted size slightly

plot_simplified_confmat(cm_roberta, axes[0], "RoBERTa External", True, custom_cmap)
plot_simplified_confmat(cm_gemma, axes[1], "Gemma External", False, custom_cmap) # Hide y-labels for the second plot

plt.tight_layout(pad=1.5) # Add padding
plt.savefig("plots/external_confusion_matrices_simplified.png", dpi=300, bbox_inches='tight')
print("Plot saved to plots/external_confusion_matrices_simplified.png")
plt.show()