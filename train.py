from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from model import get_lora_model
from config import TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, MAX_LENGTH, LEARNING_RATE, MAX_STEPS, OUTPUT_DIR
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME, APITOKEN
import numpy as np
from datasets import load_from_disk
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,token=APITOKEN, max_length=512, padding="max_length", truncation=True)

# Set a patience of 3
early_stopping_patience = 3
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)

# Load datasets
print('Loading datasets....')
train_dataset = load_from_disk("./t_data/tokenized_train")
val_dataset = load_from_disk("./t_data/tokenized_val")
print(f'Loaded dataset of length: {len(train_dataset)}')

# Load LoRA model
print('Loading LORA Model....')
peft_model = get_lora_model()
batch_size = 1

print('Define PEFT training arguments....')
# Define PEFT training arguments
peft_training_args = TrainingArguments(
    output_dir="./qwen-lora",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    learning_rate=1e-4,
    fp16=True,
    bf16=False,
    save_steps=500,
    save_total_limit=2,
    save_strategy="steps",
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=50, #500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss", #consider: try "rougeL" as metric_for_best_model
    greater_is_better=True
)

print('Initialize PEFT trainer...')
# Initialize PEFT trainer
peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[early_stopping_callback], #implement early stopping
)

print('Training PEFT model...')
# Train PEFT model
peft_trainer.train()
peft_model_path="./peft-text-summary-6.0"

print('Saving model to local directory...')
# Saving model to local directory
peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)
