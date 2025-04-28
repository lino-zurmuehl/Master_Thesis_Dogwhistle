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
plt.rcParams.update({"font.family": "Courier New "})
plt.figure(figsize=(6, 5))
plt.plot(fpr_gemma, tpr_gemma, color='#fc8d62', label=f'Gemma (AUC = {roc_auc_gemma:.2f})')
plt.plot(fpr_roberta, tpr_roberta, color='#66c2a5', label=f'RoBERTa (AUC = {roc_auc_roberta:.2f})')
plt.plot([0, 1], [0, 1], color='#8DA0CB', linestyle='--')  # Diagonal for random classifier
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison between Gemma 2 and RoBERTa')
plt.legend()
plt.tight_layout()
plt.savefig('plots/eval_roc_curve_combined.png', dpi=300, bbox_inches='tight')
plt.show()
