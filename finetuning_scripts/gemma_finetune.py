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
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import evaluate

# Model and training configuration
MODEL_ID = "google/gemma-2-2b-it"
OUTPUT_DIR = "./gemma-dogwhistle-lora"
MAX_LENGTH = 512  
BATCH_SIZE_PER_DEVICE = 8
GRAD_ACCUMULATION_STEPS = 4
LEARNING_RATE = 0.0001
EPOCHS = 3
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
WEIGHT_DECAY = 0.01
SEED = 42

# Load preprocessed training and validation data
print("Loading preprocessed data...")
train_df = pd.read_csv('gemma_train.csv')
val_df = pd.read_csv('gemma_val.csv')

# Convert pandas DataFrames to HuggingFace Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Determine the number of unique classes in the dataset
num_classes = train_df['label'].nunique()
print(f"Number of classes detected: {num_classes}")

# Compute class weights to address class imbalance in the loss function
labels_for_weights = train_df['label'].tolist()
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_for_weights),
    y=labels_for_weights
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print(f"Class weights: {class_weights}")

# Load tokenizer and set pad token if not already set
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir="/home/234533@hertie-school.lan/workspace/cache")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad_token to eos_token")

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

# Load the base model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=2,  # Binary classification
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    cache_dir="/home/234533@hertie-school.lan/workspace/cache"
)

# Ensure the model's pad_token_id is set
if model.config.pad_token_id is None:
     model.config.pad_token_id = tokenizer.pad_token_id
     print("Set model.config.pad_token_id")

# Configure LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
)

# Wrap the model with LoRA adapters
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Custom Trainer class to incorporate class weights into the loss calculation
class WeightedLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Move class weights to the correct device
        self.class_weights = class_weights.to(self.args.device)

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
    weight_decay=WEIGHT_DECAY,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False,
    fp16=False,
    bf16=True,
    logging_strategy="steps",
    logging_steps=100,
    save_total_limit=2,
    seed=SEED,
    report_to="none",
    ddp_find_unused_parameters=False,
    optim="adamw_torch_fused",
    # torch_compile=True,  # Uncomment if using PyTorch 2.0+ and want to enable torch compile
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

# Start the training process
print("Starting training...")
train_result = trainer.train()
print("Training finished.")

# Save the final model checkpoint and tokenizer
trainer.save_model(os.path.join(OUTPUT_DIR, "final_checkpoint"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_checkpoint"))
print(f"Final LoRA adapter saved to {os.path.join(OUTPUT_DIR, 'final_checkpoint')}")

# Evaluate the final model on the validation set
print("Evaluating final model...")
eval_results = trainer.evaluate()
print("Evaluation results:")
print(eval_results)

