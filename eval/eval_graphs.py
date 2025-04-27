import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Load model prediction CSVs for ROC analysis
df_gemma = pd.read_csv("eval/gemma_original_predictions.csv")  
df_roberta = pd.read_csv("eval/roberta_original_predictions.csv")

# Ensure correct data types for ROC computation
df_gemma['true_label'] = df_gemma['true_label'].astype(int)
df_gemma['probability'] = df_gemma['probability'].astype(float)
df_roberta['true_label'] = df_roberta['true_label'].astype(int)
df_roberta['probability'] = df_roberta['probability'].astype(float)

# Compute ROC curve and AUC for Gemma
fpr_gemma, tpr_gemma, _ = roc_curve(df_gemma['true_label'], df_gemma['probability'])
roc_auc_gemma = auc(fpr_gemma, tpr_gemma)

# Compute ROC curve and AUC for RoBERTa
fpr_roberta, tpr_roberta, _ = roc_curve(df_roberta['true_label'], df_roberta['probability'])
roc_auc_roberta = auc(fpr_roberta, tpr_roberta)

# Plot ROC curves for both models
sns.set(style="whitegrid", font="Courier New")
plt.figure(figsize=(6, 5))
plt.plot(fpr_gemma, tpr_gemma, color='#fc8d62', label=f'Gemma (AUC = {roc_auc_gemma:.2f})')
plt.plot(fpr_roberta, tpr_roberta, color='#66c2a5', label=f'RoBERTa (AUC = {roc_auc_roberta:.2f})')
plt.plot([0, 1], [0, 1], color='black', linestyle='--')  # Diagonal for random classifier
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

# Define color palette for confusion matrix visualization
set2 = sns.color_palette("Set2")
incorrect_color = set2[0] # Red for incorrect predictions (FP, FN)
correct_color = set2[1]   # Green for correct predictions (TN, TP)
custom_cmap = ListedColormap([incorrect_color, correct_color])

# Load external prediction results for both models
try:
    roberta_df = pd.read_csv("eval/roberta_external_predictions.csv")
    gemma_df = pd.read_csv("eval/gemma_external_predictions.csv")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure 'eval/roberta_external_predictions.csv' and 'eval/gemma_external_predictions.csv' exist.")
    exit()

# Convert label columns to integer type for confusion matrix calculation
for df in [roberta_df, gemma_df]:
    df['true_label'] = df['true_label'].astype(int)
    df['predicted_label'] = df['predicted_label'].astype(int)

# Compute confusion matrices for both models
cm_roberta = confusion_matrix(roberta_df['true_label'], roberta_df['predicted_label'])
cm_gemma = confusion_matrix(gemma_df['true_label'], gemma_df['predicted_label'])
labels = ["Negative (0)", "Positive (1)"]

def plot_simplified_confmat(cm, ax, title, show_yticklabels, cmap):
    """
    Plot a 2x2 confusion matrix with custom coloring for correct and incorrect predictions.
    Diagonal cells (correct) are colored green, off-diagonal (incorrect) are colored red.
    """
    if cm.shape != (2, 2):
        raise ValueError("This function is designed for 2x2 confusion matrices.")

    # Matrix to indicate which cells are correct (1) or incorrect (0)
    correctness_matrix = np.array([[1, 0], [0, 1]])

    sns.heatmap(
        correctness_matrix,
        annot=cm,
        fmt="d",
        cmap=cmap,
        cbar=False,
        xticklabels=labels,
        yticklabels=labels if show_yticklabels else False,
        linewidths=1,
        linecolor='gray',
        ax=ax,
        square=True,
        annot_kws={"size": 14, "color": "black"},
        vmin=0,
        vmax=1
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    if show_yticklabels:
        ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=10)

# Create side-by-side confusion matrix plots for RoBERTa and Gemma
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
plot_simplified_confmat(cm_roberta, axes[0], "RoBERTa External", True, custom_cmap)
plot_simplified_confmat(cm_gemma, axes[1], "Gemma External", False, custom_cmap)
plt.tight_layout(pad=1.5)
plt.savefig("plots/external_confusion_matrices_simplified.png", dpi=300, bbox_inches='tight')
print("Plot saved to plots/external_confusion_matrices_simplified.png")
plt.show()