import os
os.environ["HF_HOME"] = "/home/234533@hertie-school.lan/workspace/cache"
os.environ["HF_HUB_CACHE"] = "/home/234533@hertie-school.lan/workspace/cache"
os.environ["TRANSFORMERS_CACHE"] = "/home/234533@hertie-school.lan/workspace/cache"

import pandas as pd
import numpy as np
import torch
from datasets import Dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
# Note: PEFT imports are removed
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import evaluate # Use the evaluate library for metrics

# --- Configuration ---
MODEL_ID = "roberta-large" 
OUTPUT_DIR = "./roberta-dogwhistle-full-finetune" # Updated output dir name
MAX_LENGTH = 512 # Adjust based on your text length distribution and RoBERTa's limits
BATCH_SIZE_PER_DEVICE = 8 # Adjust based on GPU memory for RoBERTa-base
GRAD_ACCUMULATION_STEPS = 4 # Effective batch size = BATCH_SIZE_PER_DEVICE * NUM_GPUS * GRAD_ACCUMULATION_STEPS
LEARNING_RATE = 2e-5 # Common starting LR for full fine-tuning (adjust as needed)
EPOCHS = 4
SEED = 42
WEIGHT_DECAY = 0.0 


# --- Data Loading ---
print("Loading preprocessed data...")
train_df = pd.read_csv('roberta_train.csv')
val_df = pd.read_csv('roberta_val.csv')

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

num_classes = train_df['label'].nunique()
print(f"Number of classes detected: {num_classes}")

# Calculate class weights
labels_for_weights = train_df['label'].tolist() # Use original labels before splitting for accurate weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_for_weights),
    y=labels_for_weights
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print(f"Class weights: {class_weights}")


# --- Tokenization ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=os.environ["TRANSFORMERS_CACHE"])
# RoBERTa tokenizer usually handles padding well, explicit setting removed.

def tokenize_function(examples):
    # Tokenize the 'text' field which now contains the formatted prompt
    return tokenizer(examples["content"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

# Remove the text column after tokenization
tokenized_train_dataset = tokenized_train_dataset.remove_columns(["content"])
tokenized_val_dataset = tokenized_val_dataset.remove_columns(["content"])
tokenized_train_dataset.set_format("torch")
tokenized_val_dataset.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- Model Loading (Full Model) ---
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=num_classes, # Use the determined number of classes
    cache_dir=os.environ["TRANSFORMERS_CACHE"]
)

# --- Custom Trainer for Class Weights (Keep as is) ---
class WeightedLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    @property
    def class_weights(self):
        # Ensure weights are on the correct device right before use
        return self._class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Use CrossEntropyLoss with weights moved to the correct device
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# --- Metrics (Keep as is) ---
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Ensure predictions are valid numbers before argmax
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if np.isnan(predictions).any():
         print("Warning: NaN detected in predictions, replacing with 0")
         predictions = np.nan_to_num(predictions) # Replace NaN with zero

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


# --- Training Arguments ---
# Assumes `accelerate launch` handles device placement.
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
    per_device_eval_batch_size=BATCH_SIZE_PER_DEVICE * 2,
    gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    num_train_epochs=EPOCHS,
    evaluation_strategy="steps",
    eval_steps=100, # Evaluate every 100 steps (adjust as needed)
    save_strategy="steps",
    save_steps=100, # Save checkpoint every 100 steps (adjust as needed)
    load_best_model_at_end=True,
    metric_for_best_model=f"f1",
    push_to_hub=False,
    fp16=True,                  # Enable mixed-precision training (common for RoBERTa)
    bf16=False,                 # Disable bf16 if using fp16
    logging_strategy="steps",
    logging_steps=100,          # Log every 100 steps
    save_total_limit=2,         # Keep only the best and the latest checkpoint
    seed=SEED,
    report_to="none", 
    weight_decay=WEIGHT_DECAY,
    ddp_find_unused_parameters=False,
)

# --- Trainer Initialization ---
trainer = WeightedLossTrainer( # Use the custom trainer for class weights
    model=model, # Pass the full RoBERTa model
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --- Training ---
print("Starting full fine-tuning...")
train_result = trainer.train()
print("Training finished.")

# --- Save Final Model ---
# Saves the entire fine-tuned model
final_model_path = os.path.join(OUTPUT_DIR, "final_model")
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"Final fine-tuned model saved to {final_model_path}")

# --- Evaluation ---
print("Evaluating final model...")
eval_results = trainer.evaluate(eval_dataset=tokenized_val_dataset) # Explicitly pass dataset
print("Evaluation results:")
print(eval_results)