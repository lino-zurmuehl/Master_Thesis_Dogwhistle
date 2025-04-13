import os
# --- Environment Variables (Keep as needed) ---
os.environ["HUGGING_FACE_HUB_TOKEN"]="TOKEN" # Replace with your token if needed
os.environ["HF_HOME"] = "/home/234533@hertie-school.lan/workspace/cache"
os.environ["HF_HUB_CACHE"] = "/home/234533@hertie-school.lan/workspace/cache"
os.environ["TRANSFORMERS_CACHE"] = "/home/234533@hertie-school.lan/workspace/cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" # Let accelerate manage devices

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
MODEL_ID = "roberta-base" # Changed to RoBERTa
CSV_PATH = 'informal_reviewed.csv'
OUTPUT_DIR = "./roberta-dogwhistle-full-finetune" # Updated output dir name
MAX_LENGTH = 512 # Adjust based on your text length distribution and RoBERTa's limits
BATCH_SIZE_PER_DEVICE = 8 # Adjust based on GPU memory for RoBERTa-base
GRAD_ACCUMULATION_STEPS = 4 # Effective batch size = BATCH_SIZE_PER_DEVICE * NUM_GPUS * GRAD_ACCUMULATION_STEPS
LEARNING_RATE = 2e-5 # Common starting LR for full fine-tuning (adjust as needed)
EPOCHS = 3
SEED = 42



# --- Data Loading and Preprocessing ---
print(f"Loading data from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Ensure 'label' column exists (renaming 'lable' if necessary) and is integer type
label_column_name = 'lable'
if label_column_name not in df.columns:
    if 'label' in df.columns:
        label_column_name = 'label' # Use 'label' if 'lable' doesn't exist
    else:
        raise ValueError("CSV must contain a 'content' column and either a 'lable' or 'label' column.")
# Standardize to 'label' for Hugging Face datasets
if label_column_name != 'label':
    df.rename(columns={label_column_name: 'label'}, inplace=True)

df['label'] = df['label'].astype(int)

# Check for necessary 'content' column
if 'content' not in df.columns:
    raise ValueError("CSV must contain a 'content' column.")

# Select only the necessary columns for the dataset
df_for_dataset = df[['content', 'label']].copy()

print("Creating Hugging Face Dataset...")
# Create Hugging Face Dataset directly from the relevant columns
dataset = Dataset.from_pandas(df_for_dataset)

num_classes = df['label'].nunique()
print(f"Number of classes detected: {num_classes}")
# Cast the label column to ClassLabel
dataset = dataset.cast_column('label', ClassLabel(num_classes=num_classes))

# --- (Optional but Recommended) Calculate Class Weights ---
# Calculate weights based on the *original* full label distribution before splitting
print("Calculating class weights...")
labels_for_weights = df['label'].tolist()
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_for_weights),
    y=labels_for_weights
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print(f"Class weights: {class_weights}")

# Split data
train_val_split = dataset.train_test_split(test_size=0.1, seed=SEED, stratify_by_column="label")
train_dataset = train_val_split['train']
val_dataset = train_val_split['test']

print(f"Training on {len(train_dataset)} examples, validating on {len(val_dataset)} examples")

# Calculate class weights
labels_for_weights = df['label'].tolist() # Use original labels before splitting for accurate weights
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
    predictions = np.argmax(predictions, axis=1)

    # Determine average strategy based on number of classes
    average_strategy = "binary" if num_classes == 2 else "macro" # or "weighted"

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average=average_strategy)
    precision = precision_metric.compute(predictions=predictions, references=labels, average=average_strategy)
    recall = recall_metric.compute(predictions=predictions, references=labels, average=average_strategy)

    # For multi-class, F1, precision, recall are dictionaries if average='none'
    # We use 'macro' or 'binary' here which returns a single float
    return {
        "accuracy": accuracy["accuracy"],
        f"f1_{average_strategy}": f1["f1"],
        f"precision_{average_strategy}": precision["precision"],
        f"recall_{average_strategy}": recall["recall"],
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
    metric_for_best_model=f"f1_{'binary' if num_classes == 2 else 'macro'}", # Use appropriate metric
    push_to_hub=False,
    fp16=True,                  # Enable mixed-precision training (common for RoBERTa)
    bf16=False,                 # Disable bf16 if using fp16
    logging_strategy="steps",
    logging_steps=100,          # Log every 100 steps
    save_total_limit=2,         # Keep only the best and the latest checkpoint
    seed=SEED,
    report_to="none",           # Disable external reporting unless configured
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