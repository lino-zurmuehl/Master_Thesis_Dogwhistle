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
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import evaluate
from accelerate.utils import set_seed

# List of RoBERTa model variants to try in grid search
MODEL_IDS = ["roberta-base", "roberta-large"]
OUTPUT_DIR_BASE = "./roberta-dogwhistle-hpt"
MAX_LENGTH = 512  # Maximum sequence length for tokenization
BATCH_SIZE_PER_DEVICE = 8
GRAD_ACCUMULATION_STEPS = 4
EPOCHS = 3
SEED = 42

# Set random seed for reproducibility
set_seed(SEED)

# Load preprocessed training and validation data from CSV files
print("Loading preprocessed data...")
full_train_df = pd.read_csv('roberta_train.csv')
full_val_df = pd.read_csv('roberta_val.csv')

# Subset a fraction of the data for hyperparameter tuning (HPT)
HPT_TRAIN_SUBSET_FRACTION = 0.5
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

# Tokenization function for the 'content' column, handling missing values
def tokenize_function(examples, tokenizer):
    texts = [str(text) if text is not None else "" for text in examples["content"]]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LENGTH)

# Load evaluation metrics from the evaluate library
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

# Compute metrics for evaluation and model selection
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.array(preds) if not isinstance(preds, np.ndarray) else preds
    preds = np.nan_to_num(preds)
    preds = np.argmax(preds, axis=1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="binary")["f1"]
    prec = precision_metric.compute(predictions=preds, references=labels, average="binary")["precision"]
    rec = recall_metric.compute(predictions=preds, references=labels, average="binary")["recall"]
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}

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
        logits = outputs.logits
        num_labels = logits.size(-1)
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights_on_device)
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Define the hyperparameter grid for grid search
param_grid = {
    "model_id": MODEL_IDS,
    "learning_rate": [1e-5, 2e-5, 3e-5],
    "weight_decay": [0.0, 0.01, 0.1]
}
from itertools import product
param_combinations = [dict(zip(param_grid.keys(), vals)) for vals in product(*param_grid.values())]
print(f"Total parameter combinations to try: {len(param_combinations)}")

# Phase 1: Grid search over all parameter combinations, do not save checkpoints
grid_results = []
for idx, params in enumerate(param_combinations, 1):
    print(f"\n--- Trial {idx}/{len(param_combinations)}: {params} ---")
    set_seed(SEED + idx)

    # Load tokenizer for current model and tokenize datasets
    tokenizer = AutoTokenizer.from_pretrained(params["model_id"], cache_dir=os.environ["TRANSFORMERS_CACHE"])
    tok_train = train_dataset_hpt.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=[c for c in train_df_hpt.columns if c != 'label'])
    tok_val = val_dataset_hpt.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=[c for c in val_df_hpt.columns if c != 'label'])
    tok_train.set_format("torch"); tok_val.set_format("torch")
    data_collator = DataCollatorWithPadding(tokenizer)

    # Load the base RoBERTa model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(params["model_id"], num_labels=num_classes, cache_dir=os.environ["TRANSFORMERS_CACHE"])
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Define training arguments for HuggingFace Trainer (no checkpoint saving)
    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR_BASE, f"temp_trial_{idx}"),
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
        per_device_eval_batch_size=BATCH_SIZE_PER_DEVICE*2,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        weight_decay=params["weight_decay"],
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=False,
        report_to="none",
        seed=SEED+idx,
        ddp_find_unused_parameters=False,
    )

    # Initialize the Trainer with the custom weighted loss and metrics
    trainer = WeightedLossTrainer(
        class_weights_tensor=class_weights,
        model=model,
        args=args,
        train_dataset=tok_train,
        eval_dataset=tok_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    grid_results.append({
        'trial': idx,
        'model_id': params['model_id'],
        'learning_rate': params['learning_rate'],
        'weight_decay': params['weight_decay'],
        'f1': metrics['eval_f1'],
        'accuracy': metrics['eval_accuracy'],
        'precision': metrics['eval_precision'],
        'recall': metrics['eval_recall'],
    })

    # Free memory after each trial
    del model, trainer; gc.collect(); torch.cuda.empty_cache()

# Save grid search results to CSV for later analysis
results_df = pd.DataFrame(grid_results)
results_df.to_csv(os.path.join(OUTPUT_DIR_BASE, "grid_search_results.csv"), index=False)
print("Grid search completed. Results saved.")

# Phase 2: Retrain and save only the top 3 trials based on F1 score
top3 = results_df.sort_values('f1', ascending=False).head(3)
final_results = []
for rank, row in enumerate(top3.itertuples(index=False), start=1):
    params = {"model_id": row.model_id, "learning_rate": row.learning_rate, "weight_decay": row.weight_decay}
    print(f"\n=== Saving top {rank}: {params} ===")

    set_seed(SEED + int(row.trial))
    tokenizer = AutoTokenizer.from_pretrained(params["model_id"], cache_dir=os.environ["TRANSFORMERS_CACHE"])
    tok_train = train_dataset_hpt.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=[c for c in train_df_hpt.columns if c != 'label'])
    tok_val = val_dataset_hpt.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=[c for c in val_df_hpt.columns if c != 'label'])
    tok_train.set_format("torch"); tok_val.set_format("torch")
    data_collator = DataCollatorWithPadding(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(params["model_id"], num_labels=num_classes, cache_dir=os.environ["TRANSFORMERS_CACHE"])
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    output_dir = os.path.join(OUTPUT_DIR_BASE, f"best_trial_{rank}")
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
        per_device_eval_batch_size=BATCH_SIZE_PER_DEVICE*2,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        weight_decay=params["weight_decay"],
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        seed=SEED+int(row.trial),
        ddp_find_unused_parameters=False,
    )

    trainer = WeightedLossTrainer(
        class_weights_tensor=class_weights,
        model=model,
        args=args,
        train_dataset=tok_train,
        eval_dataset=tok_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    final_results.append({**params, 'f1': metrics['eval_f1'], 'accuracy': metrics['eval_accuracy'], 'precision': metrics['eval_precision'], 'recall': metrics['eval_recall']})

    # Free memory after each retraining
    del model, trainer; gc.collect(); torch.cuda.empty_cache()

# Save results of top 3 retrained models
pd.DataFrame(final_results).to_csv(os.path.join(OUTPUT_DIR_BASE, "top3_saved_results.csv"), index=False)
print("Top 3 models trained and saved.")
