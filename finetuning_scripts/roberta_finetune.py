import os
# Set HuggingFace cache directories for model and tokenizer downloads
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
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import evaluate

# Model and training configuration
MODEL_ID = "roberta-large"
OUTPUT_DIR = "./roberta-dogwhistle-full-finetune"
MAX_LENGTH = 512
BATCH_SIZE_PER_DEVICE = 8
GRAD_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
EPOCHS = 4
SEED = 42
WEIGHT_DECAY = 0.0

# Load preprocessed training and validation data from CSV files
print("Loading preprocessed data...")
train_df = pd.read_csv('roberta_train.csv')
val_df = pd.read_csv('roberta_val.csv')

# Convert pandas DataFrames to HuggingFace Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Determine the number of unique classes in the dataset
num_classes = train_df['label'].nunique()
print(f"Number of classes detected: {num_classes}")

# Compute class weights for imbalanced classification
labels_for_weights = train_df['label'].tolist()
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_for_weights),
    y=labels_for_weights
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print(f"Class weights: {class_weights}")

# Load tokenizer for RoBERTa model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=os.environ["TRANSFORMERS_CACHE"])

# Tokenization function for the 'content' column
def tokenize_function(examples):
    return tokenizer(examples["content"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

# Apply tokenization to the datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

# Remove the original text column after tokenization and set format for PyTorch
tokenized_train_dataset = tokenized_train_dataset.remove_columns(["content"])
tokenized_val_dataset = tokenized_val_dataset.remove_columns(["content"])
tokenized_train_dataset.set_format("torch")
tokenized_val_dataset.set_format("torch")

# Data collator for dynamic padding during batching
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load the base RoBERTa model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=num_classes,
    cache_dir=os.environ["TRANSFORMERS_CACHE"]
)

# Custom Trainer class to incorporate class weights into the loss calculation
class WeightedLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    @property
    def class_weights(self):
        # Move class weights to the correct device before use
        return self._class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute weighted cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Load evaluation metrics from the evaluate library
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

# Compute metrics for evaluation and model selection
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Ensure predictions are valid numbers before argmax
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

# Define training arguments for HuggingFace Trainer
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
    per_device_eval_batch_size=BATCH_SIZE_PER_DEVICE * 2,
    gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    num_train_epochs=EPOCHS,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False,
    fp16=True,
    bf16=False,
    logging_strategy="steps",
    logging_steps=100,
    save_total_limit=2,
    seed=SEED,
    report_to="none",
    weight_decay=WEIGHT_DECAY,
    ddp_find_unused_parameters=False,
)

# Initialize the Trainer with the custom weighted loss and metrics
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start the full fine-tuning process
print("Starting full fine-tuning...")
train_result = trainer.train()
print("Training finished.")

# Save the final fine-tuned model and tokenizer
final_model_path = os.path.join(OUTPUT_DIR, "final_model")
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"Final fine-tuned model saved to {final_model_path}")

# Evaluate the final model on the validation set
print("Evaluating final model...")
eval_results = trainer.evaluate(eval_dataset=tokenized_val_dataset)
print("Evaluation results:")
print(eval_results)