import os
# Set HuggingFace cache directories for model and tokenizer downloads
os.environ["HF_HOME"] = "/home/234533@hertie-school.lan/workspace/cache"
os.environ["HF_HUB_CACHE"] = "/home/234533@hertie-school.lan/workspace/cache"
os.environ["TRANSFORMERS_CACHE"] = "/home/234533@hertie-school.lan/workspace/cache"

import pandas as pd
import numpy as np
import gc
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import evaluate
from accelerate.utils import set_seed

# Model and experiment configuration
MODEL_ID = "google/gemma-2-2b-it"
OUTPUT_DIR_BASE = "./gemma-dogwhistle-lora-hpt"
MAX_LENGTH = 512  
BATCH_SIZE_PER_DEVICE = 4
GRAD_ACCUMULATION_STEPS = 4
EPOCHS = 2
SEED = 42
LORA_DROPOUT = 0.05
LORA_ALPHA = 32

# Set cache directories again for redundancy (can be removed if not needed)
os.environ["HF_HOME"] = "/home/234533@hertie-school.lan/workspace/cache"
os.environ["HF_HUB_CACHE"] = "/home/234533@hertie-school.lan/workspace/cache"
os.environ["TRANSFORMERS_CACHE"] = "/home/234533@hertie-school.lan/workspace/cache"

# Set random seed for reproducibility
set_seed(SEED)

# Load preprocessed training and validation data from CSV files
print("Loading preprocessed data...")
full_train_df = pd.read_csv('gemma_train.csv')
full_val_df = pd.read_csv('gemma_val.csv')

# Subset a fraction of the data for hyperparameter tuning (HPT)
HPT_TRAIN_SUBSET_FRACTION = 0.3
HPT_VAL_SUBSET_FRACTION = 0.5

_, train_df_hpt = train_test_split(
    full_train_df, 
    test_size=HPT_TRAIN_SUBSET_FRACTION, 
    random_state=SEED, 
    stratify=full_train_df['label']
)
_, val_df_hpt = train_test_split(
    full_val_df, 
    test_size=HPT_VAL_SUBSET_FRACTION, 
    random_state=SEED, 
    stratify=full_val_df['label']
)

print(f"Using {len(train_df_hpt)} samples for HPT training subset.")
print(f"Using {len(val_df_hpt)} samples for HPT validation subset.")

# Convert pandas DataFrames to HuggingFace Datasets
train_dataset_hpt = Dataset.from_pandas(train_df_hpt)
val_dataset_hpt = Dataset.from_pandas(val_df_hpt)

# Determine the number of unique classes in the dataset
num_classes = full_train_df['label'].nunique()
print(f"Number of classes detected: {num_classes}")

# Compute class weights for imbalanced classification
labels_for_weights = full_train_df['label'].tolist()
class_weights_np = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_for_weights),
    y=labels_for_weights
)
class_weights = torch.tensor(class_weights_np, dtype=torch.float)
print(f"Class weights: {class_weights}")

# Load tokenizer and set pad token if not already set
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    cache_dir="/home/234533@hertie-school.lan/workspace/cache"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad_token to eos_token")

# Tokenization function for the 'content' column, handling missing values
def tokenize_function(examples):
    texts = [str(text) if text is not None else "" for text in examples["content"]]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LENGTH)

# Tokenize the HPT subsets and remove all columns except 'label'
tokenized_train_dataset_hpt = train_dataset_hpt.map(
    tokenize_function, batched=True, 
    remove_columns=[col for col in train_df_hpt.columns if col != 'label']
)
tokenized_val_dataset_hpt = val_dataset_hpt.map(
    tokenize_function, batched=True, 
    remove_columns=[col for col in val_df_hpt.columns if col != 'label']
)
tokenized_train_dataset_hpt.set_format("torch")
tokenized_val_dataset_hpt.set_format("torch")

# Data collator for dynamic padding during batching
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load evaluation metrics from the evaluate library
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

# Compute metrics for evaluation and model selection
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if np.isnan(predictions).any():
         print("Warning: NaN detected in predictions, replacing with 0")
         predictions = np.nan_to_num(predictions)
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")
    precision = precision_metric.compute(predictions=predictions, references=labels, average="binary")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="binary")
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"],
    }

# Custom Trainer class to incorporate class weights into the loss calculation
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights_tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights_tensor = class_weights_tensor
        self._weights_moved = False

    def compute_loss(self, model, inputs, return_outputs=False):
        if not self._weights_moved and self.args.device:
             self.class_weights_on_device = self.class_weights_tensor.to(self.args.device)
             self._weights_moved = True
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if not hasattr(self, 'class_weights_on_device'):
             self.class_weights_on_device = self.class_weights_tensor.to(logits.device)
             self._weights_moved = True
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights_on_device)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Define the hyperparameter grid for grid search
param_grid = {
    "learning_rate": [1e-5, 3e-5, 1e-4],
    "lora_r": [8, 16, 32],
    "weight_decay": [0.01, 0.1]
}

# Generate all combinations of hyperparameters from the grid
def get_param_combinations(param_grid):
    """Generate all combinations of parameters from the grid"""
    from itertools import product
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(product(*values))
    return [dict(zip(keys, combo)) for combo in combinations]

param_combinations = get_param_combinations(param_grid)
print(f"Total parameter combinations to try: {len(param_combinations)}")

# Run grid search over all parameter combinations and collect results
results = []

for idx, params in enumerate(param_combinations):
    print(f"\n--- Starting Trial {idx+1}/{len(param_combinations)} ---")
    print(params)
    
    # Set seed for reproducibility for each trial
    set_seed(SEED + idx)
    
    # Output directory for this trial
    trial_output_dir = os.path.join(OUTPUT_DIR_BASE, f"grid_trial_{idx+1}")
    
    # Load the base model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=num_classes,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir="/home/234533@hertie-school.lan/workspace/cache"
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Configure LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Define training arguments for HuggingFace Trainer
    training_args = TrainingArguments(
        output_dir=trial_output_dir,
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
        per_device_eval_batch_size=BATCH_SIZE_PER_DEVICE * 2,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        weight_decay=params["weight_decay"],
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
        fp16=False,
        bf16=True,
        logging_strategy="steps",
        logging_steps=20,
        save_total_limit=1,
        seed=SEED + idx,
        report_to="none",
        ddp_find_unused_parameters=False,
        warmup_steps=50,
    )
    
    # Initialize the Trainer with the custom weighted loss and metrics
    trainer = WeightedLossTrainer(
        class_weights_tensor=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset_hpt,
        eval_dataset=tokenized_val_dataset_hpt,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train and evaluate the model for this hyperparameter combination
    try:
        trainer.train()
        eval_results = trainer.evaluate()
        
        # Store results for this trial
        trial_results = {
            "trial_id": idx + 1,
            "f1": eval_results["eval_f1"],
            "accuracy": eval_results["eval_accuracy"],
            "precision": eval_results["eval_precision"],
            "recall": eval_results["eval_recall"],
            **params
        }
        results.append(trial_results)
        print(f"Trial {idx+1} results: {eval_results}")
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error in trial {idx+1}: {e}")
        # Add failed trial with NaN metrics and error message
        results.append({
            "trial_id": idx + 1,
            "f1": float('nan'),
            "accuracy": float('nan'),
            "precision": float('nan'),
            "recall": float('nan'),
            **params,
            "error": str(e)
        })

# Save all grid search results to a CSV file for later analysis
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR_BASE, "grid_search_results.csv"), index=False)
print(f"Grid search results saved to {os.path.join(OUTPUT_DIR_BASE, 'grid_search_results.csv')}")

# Find and display the best hyperparameter combination based on F1 score
if not results_df.empty and not results_df["f1"].isna().all():
    best_idx = results_df["f1"].idxmax()
    best_params = results_df.iloc[best_idx].to_dict()
    print("\n--- Best Parameters Found ---")
    print(f"F1 Score: {best_params['f1']:.4f}")
    print("Parameters:")
    for key, value in best_params.items():
        if key not in ["trial_id", "f1", "accuracy", "precision", "recall"]:
            print(f"  {key}: {value}")
else:
    print("No valid results found.")

