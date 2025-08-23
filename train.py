from transformers import TrainingArguments, Trainer
from dataset import load_cnn_dailymail
from model import get_lora_model
from config import TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, MAX_LENGTH, LEARNING_RATE, MAX_STEPS, OUTPUT_DIR
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME, APITOKEN

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,token=APITOKEN, max_length=512, padding="max_length", truncation=True)

# Load datasets
print('Loading datasets....')
train_dataset, val_dataset = load_cnn_dailymail()
print(f'Loaded dataset of length: {len(train_dataset)}')

# Load LoRA model
print('Loading LORA Model....')
peft_model = get_lora_model()
batch_size = 128

print('Define PEFT training arguments....')
# Define PEFT training arguments
peft_training_args = TrainingArguments(
    output_dir="./qwen-lora",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    learning_rate=3e-4,
    fp16=False,
    bf16=False,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
)

print('Initialize PEFT trainer...')
# Initialize PEFT trainer
peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print('Training PEFT model...')
# Train PEFT model
peft_trainer.train()
peft_model_path="./peft-text-summary"

print('Saving model to local directory...')
# Saving model to local directory
peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)
