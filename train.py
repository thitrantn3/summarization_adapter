from transformers import TrainingArguments
from dataset import load_cnn_dailymail
from model import get_lora_model
from config import TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, MAX_LENGTH, LEARNING_RATE, MAX_STEPS, OUTPUT_DIR
from peft import LoraConfig, get_peft_model, PEFTTrainer

# Load datasets
train_dataset, val_dataset = load_cnn_dailymail()

# Load LoRA model
model = get_lora_model()

# Define training arguments
training_args = TrainingArguments(
    output_dir="./qwen-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=3e-4,
    fp16=True,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
)

# Initialize PEFT trainer
trainer = PEFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    args=training_args
)
#Start training
trainer.train()