"""
SFT: LoRA fine-tuning of Qwen2.5-3B on CNN/DailyMail (run dataset.py first).

    python dataset.py
    python train.py                      # saves the adapter to SFT_OUTPUT (config.py)
"""
import math

import torch
from datasets import load_from_disk
from transformers import (AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback,
                          Trainer, TrainingArguments)

from config import (APITOKEN, EARLY_STOPPING_PATIENCE, EVAL_BATCH_SIZE, EVAL_STEPS, GRAD_ACCUM,
                    LEARNING_RATE, MODEL_NAME, NUM_EPOCHS, OUTPUT_DIR, SFT_OUTPUT, TRAIN_BATCH_SIZE)
from model import get_lora_model, train_dtype

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=APITOKEN)
tokenizer.pad_token = tokenizer.eos_token

print('Loading datasets....')
train_dataset = load_from_disk("./t_data/tokenized_train")
val_dataset = load_from_disk("./t_data/tokenized_val")
if "labels" not in train_dataset.column_names or -100 not in train_dataset[0]["labels"]:
    raise SystemExit("t_data is in the old format (loss on the whole text) - rerun `python dataset.py`")
print(f'Loaded {len(train_dataset)} train / {len(val_dataset)} val examples')

print('Loading LORA Model....')
peft_model = get_lora_model()
peft_model.print_trainable_parameters()

# Pads input_ids with pad_token and labels with -100 per batch (no wasted padding, no loss on it)
collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8)

dtype = train_dtype()
steps = math.ceil(len(train_dataset) / (TRAIN_BATCH_SIZE * GRAD_ACCUM)) * NUM_EPOCHS
print('Define PEFT training arguments....')
peft_training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=max(1, math.ceil(0.05 * steps)),  # 5%; warmup_ratio was removed in newer transformers
    bf16=dtype == torch.bfloat16,
    fp16=dtype == torch.float16,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=EVAL_STEPS,          # must line up with eval_steps for load_best_model_at_end
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,        # lower loss is better (was True: kept the worst checkpoint)
    report_to="none",
)

print('Initialize PEFT trainer...')
peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)

print(f'Training PEFT model ({steps} steps max)...')
peft_trainer.train()

print(f'Saving model to {SFT_OUTPUT}...')
peft_trainer.model.save_pretrained(SFT_OUTPUT)
tokenizer.save_pretrained(SFT_OUTPUT)
