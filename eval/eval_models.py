import os
os.environ["HF_HOME"] = "/home/234533@hertie-school.lan/workspace/cache"
os.environ["HF_HUB_CACHE"] = "/home/234533@hertie-school.lan/workspace/cache"
os.environ["TRANSFORMERS_CACHE"] = "/home/234533@hertie-school.lan/workspace/cache"

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
from accelerate import Accelerator
from typing import List, Tuple, Dict, Any

BATCH_SIZE = 16  # Number of samples per batch for inference
MAX_LENGTH = 512 

def load_model_and_tokenizer(model_path: str, cache_dir: str, accelerator: Accelerator):
    """Load a HuggingFace model and tokenizer, move model to the correct device."""
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    print(f"Loading model from: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir
    )
    print("Preparing model with Accelerator...")
    model = accelerator.prepare_model(model)
    print("Model and tokenizer loaded.")
    return model, tokenizer

def predict_batch_probabilities(texts: List[str], model, tokenizer, accelerator: Accelerator) -> List[float]:
    """Tokenize a batch of texts, run through model, and return positive class probabilities."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=-1)
        positive_probs = probs[:, 1].cpu().tolist()
    return positive_probs

def evaluate_model(
    model,
    tokenizer,
    texts: List[str],
    labels: List[int],
    model_name: str,
    accelerator: Accelerator,
    desc: str = "Evaluating"
) -> Tuple[List[int], List[float]]:
    """Run model inference in batches, return predicted labels and probabilities."""
    predictions = []
    probabilities = []
    model.eval()
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"{desc} (Batch Size: {BATCH_SIZE})", unit="batch"):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_probs = predict_batch_probabilities(batch_texts, model, tokenizer, accelerator)
        batch_preds = [1 if prob >= 0.5 else 0 for prob in batch_probs]
        predictions.extend(batch_preds)
        probabilities.extend(batch_probs)
    return predictions, probabilities

def load_original_dataset(model_name: str) -> pd.DataFrame:
    """Load the validation set for the given model (Gemma or RoBERTa)."""
    if 'gemma' in model_name.lower():
        print("Loading Gemma validation dataset (gemma_val.csv)...")
        file_path = 'gemma_val.csv'
    else:
        print("Loading RoBERTa validation dataset (roberta_val.csv)...")
        file_path = 'roberta_val.csv'
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}")
        raise

def load_external_dataset(cache_dir: str) -> pd.DataFrame | None:
    """Load the external SALT-NLP/silent_signals_detection dataset from HuggingFace."""
    print("Loading external dataset (SALT-NLP/silent_signals_detection) from Hugging Face...")
    try:
        hf_dataset = load_dataset("SALT-NLP/silent_signals_detection", cache_dir=cache_dir)
        if 'train' not in hf_dataset:
            print("Error: 'train' split not found in the external dataset.")
            return None
        external_df = pd.DataFrame(hf_dataset['train'])
        print(f"External dataset columns: {external_df.columns.tolist()}")
        if 'label' not in external_df.columns:
            print("Error: 'label' column not found in the external dataset.")
            return None
        external_df['label'] = external_df['label'].apply(lambda x: 1 if x == 'coded' else 0)
        if 'text' in external_df.columns:
            external_df.rename(columns={'text': 'content'}, inplace=True)
        elif 'content' not in external_df.columns:
            print("Error: Neither 'text' nor 'content' column found in the external dataset.")
            return None
        print("External dataset loaded and processed successfully.")
        return external_df
    except Exception as e:
        print(f"Error loading or processing external dataset: {str(e)}")
        return None

def plot_roc_curve(true_labels: List[int], probabilities: List[float], model_name: str, dataset_name: str, output_path: str | None = None):
    """Plot and optionally save the ROC curve for a model on a dataset."""
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {model_name} on {dataset_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    if output_path:
        print(f"Saving ROC curve to: {output_path}")
        plt.savefig(output_path)
    plt.close()
    return roc_auc

def plot_combined_roc_curves(results: Dict[str, Dict[str, Any]], dataset_name: str, output_path: str | None = None):
    """Plot ROC curves for all models on a dataset in a single figure."""
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    for model_name, result in results.items():
        if 'true_labels' in result and 'probabilities' in result:
            fpr, tpr, _ = roc_curve(result['true_labels'], result['probabilities'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
        else:
            print(f"Warning: Missing data for ROC curve for model '{model_name}' on {dataset_name}.")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Comparison of ROC Curves on {dataset_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    if output_path:
        print(f"Saving combined ROC curves to: {output_path}")
        plt.savefig(output_path)
    plt.close()

def main():
    # Initialize accelerator for device management (CPU/GPU/TPU)
    accelerator = Accelerator()
    print(f"Using device: {accelerator.device}")

    # Model checkpoints (update these paths as needed)
    models = {
        'RoBERTa': './roberta-dogwhistle-full-finetune/final_model',
        'Gemma': './gemma-dogwhistle-lora/checkpoint-1400'
    }

    # Load the external validation dataset once for all models
    external_df = load_external_dataset(CACHE_DIR)

    # Dictionaries to store evaluation results for each model
    original_results = {}
    external_results = {}

    for model_name, model_path in models.items():
        print(f"\n{'-'*50}")
        print(f"Starting evaluation for {model_name} model ({model_path})...")
        print(f"{'-'*50}")

        try:
            # Load model and tokenizer for current model
            model, tokenizer = load_model_and_tokenizer(model_path, CACHE_DIR, accelerator)

            # --- Evaluate on original validation set ---
            print(f"\n--- Evaluating {model_name} on Original Dataset ---")
            original_df = load_original_dataset(model_name)
            if 'content' not in original_df.columns or 'label' not in original_df.columns:
                print(f"Error: Original dataset for {model_name} lacks 'content' or 'label' column.")
                continue
            original_texts = original_df['content'].astype(str).tolist()
            original_labels = original_df['label'].astype(int).tolist()
            original_preds, original_probs = evaluate_model(
                model, tokenizer, original_texts, original_labels, model_name,
                accelerator, f"Evaluating {model_name} on original data"
            )
            original_accuracy = accuracy_score(original_labels, original_preds)
            original_report = classification_report(original_labels, original_preds, zero_division=0)
            original_results[model_name] = {
                'accuracy': original_accuracy,
                'classification_report': original_report,
                'predictions': original_preds,
                'probabilities': original_probs,
                'true_labels': original_labels
            }
            print(f"\n{model_name} Results on Original Dataset:")
            print(f"Accuracy: {original_accuracy:.4f}")
            print("\nClassification Report:")
            print(original_report)
            original_pred_df = pd.DataFrame({
                'text': original_texts,
                'true_label': original_labels,
                'predicted_label': original_preds,
                'probability': original_probs
            })
            original_pred_output_path = f'{model_name.lower()}_original_predictions.csv'
            original_pred_df.to_csv(original_pred_output_path, index=False)
            print(f"Saved original predictions to {original_pred_output_path}")
            original_roc_auc = plot_roc_curve(
                original_labels, original_probs, model_name,
                'Original Dataset', f'{model_name.lower()}_original_roc.png'
            )
            print(f"{model_name} Original Dataset ROC AUC: {original_roc_auc:.4f}")

            # --- Evaluate on external validation set ---
            print(f"\n--- Evaluating {model_name} on External Dataset ---")
            if external_df is not None:
                external_texts = external_df['content'].astype(str).tolist()
                external_labels = external_df['label'].astype(int).tolist()
                external_preds, external_probs = evaluate_model(
                    model, tokenizer, external_texts, external_labels, model_name,
                    accelerator, f"Evaluating {model_name} on external data"
                )
                external_accuracy = accuracy_score(external_labels, external_preds)
                external_report = classification_report(external_labels, external_preds, zero_division=0)
                external_results[model_name] = {
                    'accuracy': external_accuracy,
                    'classification_report': external_report,
                    'predictions': external_preds,
                    'probabilities': external_probs,
                    'true_labels': external_labels
                }
                print(f"\n{model_name} Results on External Dataset:")
                print(f"Accuracy: {external_accuracy:.4f}")
                print("\nClassification Report:")
                print(external_report)
                external_pred_df = pd.DataFrame({
                    'text': external_texts,
                    'true_label': external_labels,
                    'predicted_label': external_preds,
                    'probability': external_probs
                })
                external_pred_output_path = f'{model_name.lower()}_external_predictions.csv'
                external_pred_df.to_csv(external_pred_output_path, index=False)
                print(f"Saved external predictions to {external_pred_output_path}")
                external_roc_auc = plot_roc_curve(
                    external_labels, external_probs, model_name,
                    'External Dataset', f'{model_name.lower()}_external_roc.png'
                )
                print(f"{model_name} External Dataset ROC AUC: {external_roc_auc:.4f}")
            else:
                print(f"Skipping evaluation on external dataset for {model_name} as it failed to load.")

            # Free up memory after each model
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except FileNotFoundError as e:
            print(f"Error: Required file not found during {model_name} evaluation: {e}. Skipping this model.")
            continue
        except Exception as e:
            print(f"An unexpected error occurred evaluating {model_name} model: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Plot ROC curves for all models on both datasets
    print("\n--- Generating Combined ROC Curves ---")
    if original_results:
        plot_combined_roc_curves(original_results, 'Original Dataset', 'combined_original_roc_curves.png')
    else:
        print("No results available to generate combined ROC for original datasets.")
    if external_results:
        plot_combined_roc_curves(external_results, 'External Dataset', 'combined_external_roc_curves.png')
    else:
        print("No results available to generate combined ROC for the external dataset.")

    # Write a summary report with accuracy, classification report, and ROC AUC for each model
    print("\n--- Saving Evaluation Summary ---")
    summary_output_path = 'evaluation_summary_report.txt'
    try:
        with open(summary_output_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write(" EVALUATION SUMMARY REPORT\n")
            f.write("="*60 + "\n\n")
            if original_results:
                f.write("--- EVALUATION ON ORIGINAL DATASETS ---\n\n")
                for model_name, result in original_results.items():
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Accuracy: {result['accuracy']:.4f}\n")
                    f.write("Classification Report:\n")
                    f.write(result['classification_report'])
                    if 'true_labels' in result and 'probabilities' in result:
                        fpr, tpr, _ = roc_curve(result['true_labels'], result['probabilities'])
                        roc_auc = auc(fpr, tpr)
                        f.write(f"\nROC AUC Score: {roc_auc:.4f}\n")
                    f.write("\n" + "-"*50 + "\n")
            else:
                f.write("--- No results recorded for original datasets. ---\n\n")
            if external_results:
                f.write("\n\n--- EVALUATION ON EXTERNAL DATASET ---\n\n")
                for model_name, result in external_results.items():
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Accuracy: {result['accuracy']:.4f}\n")
                    f.write("Classification Report:\n")
                    f.write(result['classification_report'])
                    if 'true_labels' in result and 'probabilities' in result:
                        fpr, tpr, _ = roc_curve(result['true_labels'], result['probabilities'])
                        roc_auc = auc(fpr, tpr)
                        f.write(f"\nROC AUC Score: {roc_auc:.4f}\n")
                    f.write("\n" + "-"*50 + "\n")
            else:
                f.write("\n\n--- No results recorded for the external dataset. ---\n")
        print(f"Saved evaluation summary to {summary_output_path}")
    except Exception as e:
        print(f"Error saving summary report: {e}")

    print("\nEvaluation script finished.")

if __name__ == "__main__":
    main()