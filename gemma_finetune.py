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
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
import evaluate # Use the evaluate library for metrics

# --- Configuration ---
MODEL_ID = "google/gemma-2-2b-it" 
OUTPUT_DIR = "./gemma-dogwhistle-lora"
MAX_LENGTH = 512 # Adjust based on your text length distribution
BATCH_SIZE_PER_DEVICE = 8 
GRAD_ACCUMULATION_STEPS = 4 
LEARNING_RATE = 0.0001
EPOCHS = 3
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
WEIGHT_DECAY = 0.01
SEED = 42

# --- Data Loading ---
print("Loading preprocessed data...")
train_df = pd.read_csv('gemma_train.csv')
val_df = pd.read_csv('gemma_val.csv')

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
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir="/home/234533@hertie-school.lan/workspace/cache")
# Gemma specific: often requires explicit pad token setting, though newer versions might handle it.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad_token to eos_token")

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

# --- Model Loading and LoRA Configuration ---
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=2, # Binary classification (0 or 1)
    torch_dtype=torch.bfloat16, # Use bfloat16 on A100s
    trust_remote_code=True, # Might be needed depending on the exact Gemma version/HF implementation
    cache_dir="/home/234533@hertie-school.lan/workspace/cache"
)
# Set pad_token_id for the model config if needed
if model.config.pad_token_id is None:
     model.config.pad_token_id = tokenizer.pad_token_id
     print("Set model.config.pad_token_id")


peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, # Sequence Classification
    inference_mode=False,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    # Target modules might need adjustment based on the specific Gemma model version
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

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --- Custom Trainer for Class Weights ---
class WeightedLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Move weights to the device the model is on
        self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Use CrossEntropyLoss with weights
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# --- Metrics ---
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary") # Adjust average if needed
    precision = precision_metric.compute(predictions=predictions, references=labels, average="binary")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="binary")

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"],
    }

# --- Training Arguments ---
# Assumes you are using accelerate for multi-GPU. `accelerate` handles device placement.
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
    load_best_model_at_end=True, # Load the best model based on eval metric at the end
    metric_for_best_model="f1", # Choose metric to determine the best model (e.g., f1 or accuracy)
    push_to_hub=False,          
    fp16=False,                 
    bf16=True,                  
    logging_strategy="steps",
    logging_steps=100,          # Log less frequently
    save_total_limit=2,        # Keep only the best and the latest checkpoint
    seed=SEED,
    report_to="none",  
    ddp_find_unused_parameters=False,  
    optim="adamw_torch_fused", # Use fused AdamW if available (good on newer GPUs) - requires testing
    # torch_compile=True,       # Optional: Use torch compile for potential speedup (requires PyTorch 2.0+) - requires testing
)

# --- Trainer Initialization ---
trainer = WeightedLossTrainer( # Use the custom trainer
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --- Training ---
print("Starting training...")
train_result = trainer.train()
print("Training finished.")

# --- Save Final Model ---
trainer.save_model(os.path.join(OUTPUT_DIR, "final_checkpoint")) # Save the final adapter
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_checkpoint"))
print(f"Final LoRA adapter saved to {os.path.join(OUTPUT_DIR, 'final_checkpoint')}")

# --- Evaluation ---
print("Evaluating final model...")
eval_results = trainer.evaluate()
print("Evaluation results:")
print(eval_results)

