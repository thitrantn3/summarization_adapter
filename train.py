"""
SFT: LoRA fine-tuning of Qwen2.5-3B on CNN/DailyMail (run dataset.py first).

    python dataset.py
    python train.py                      # saves the adapter to SFT_OUTPUT (config.py)

Style warm-up before DPO: a short SFT pass on the *chosen* side of a DPO .jsonl
(prompt/chosen columns), continuing from an existing SFT adapter, so the model already
writes in the target style when DPO starts. Loss is only on the summary + <eos>.

    python train.py --jsonl dpo_data/genz/train.jsonl          # -> ./peft-genz-sft
    python train_dpo.py --sft_path ./peft-genz-sft \
        --train_file dpo_data/genz/train.jsonl --val_file dpo_data/genz/val.jsonl

  --val_jsonl      defaults to val.jsonl next to --jsonl
  --init_adapter   adapter to continue from (default ./peft-text-summary, what DPO used so far;
                   pass "" to start a fresh LoRA)
  --output, --epochs, --lr   warm-up defaults: ./peft-genz-sft, 1 epoch, 5e-5
"""
import argparse
import math
import os

import torch
from datasets import load_from_disk
from transformers import (AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback,
                          Trainer, TrainingArguments)

from config import (APITOKEN, EARLY_STOPPING_PATIENCE, EVAL_BATCH_SIZE, EVAL_STEPS, GRAD_ACCUM,
                    LEARNING_RATE, MODEL_NAME, NUM_EPOCHS, OUTPUT_DIR, SFT_OUTPUT, TRAIN_BATCH_SIZE)
from model import get_lora_model, train_dtype

ap = argparse.ArgumentParser()
ap.add_argument("--jsonl", help="style warm-up: train on prompt/chosen from this .jsonl instead of t_data")
ap.add_argument("--val_jsonl", help="validation .jsonl (default: val.jsonl next to --jsonl)")
ap.add_argument("--completion_col", default="chosen")
ap.add_argument("--init_adapter", default=None,
                help="continue training this adapter (warm-up default ./peft-text-summary; '' = fresh LoRA)")
ap.add_argument("--output", default=None, help=f"where to save the adapter (default {SFT_OUTPUT}, warm-up ./peft-genz-sft)")
ap.add_argument("--epochs", type=float, default=None, help=f"default {NUM_EPOCHS}, warm-up 1")
ap.add_argument("--lr", type=float, default=None, help=f"default {LEARNING_RATE}, warm-up 5e-5")
args = ap.parse_args()

warmup = bool(args.jsonl)
init_adapter = args.init_adapter if args.init_adapter is not None else ("./peft-text-summary" if warmup else "")
output = args.output or ("./peft-genz-sft" if warmup else SFT_OUTPUT)
epochs = args.epochs or (1 if warmup else NUM_EPOCHS)
lr = args.lr or (5e-5 if warmup else LEARNING_RATE)
ckpt_dir = OUTPUT_DIR + "-warmup" if warmup else OUTPUT_DIR

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=APITOKEN)
tokenizer.pad_token = tokenizer.eos_token

print('Loading datasets....')
if warmup:
    from dataset import load_jsonl_sft
    val_path = args.val_jsonl or os.path.join(os.path.dirname(args.jsonl), "val.jsonl")
    if not os.path.isfile(val_path):
        raise SystemExit(f"no validation file {val_path}; pass --val_jsonl")
    train_dataset = load_jsonl_sft(args.jsonl, tokenizer, args.completion_col)
    val_dataset = load_jsonl_sft(val_path, tokenizer, args.completion_col)
else:
    train_dataset = load_from_disk("./t_data/tokenized_train")
    val_dataset = load_from_disk("./t_data/tokenized_val")
    if "labels" not in train_dataset.column_names or -100 not in train_dataset[0]["labels"]:
        raise SystemExit("t_data is in the old format (loss on the whole text) - rerun `python dataset.py`")
print(f'Loaded {len(train_dataset)} train / {len(val_dataset)} val examples')
if init_adapter and not os.path.isdir(init_adapter):
    raise SystemExit(f"--init_adapter {init_adapter} not found (pass '' to start a fresh LoRA)")
print(f'epochs {epochs}, lr {lr}, init adapter {init_adapter or "none (fresh LoRA)"}, output {output}')

print('Loading LORA Model....')
peft_model = get_lora_model(init_adapter or None)
peft_model.print_trainable_parameters()

# Pads input_ids with pad_token and labels with -100 per batch (no wasted padding, no loss on it)
collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8)

dtype = train_dtype()
steps = math.ceil(math.ceil(len(train_dataset) / (TRAIN_BATCH_SIZE * GRAD_ACCUM)) * epochs)
print('Define PEFT training arguments....')
peft_training_args = TrainingArguments(
    output_dir=ckpt_dir,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=epochs,
    learning_rate=lr,
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

print(f'Saving model to {output}...')
peft_trainer.model.save_pretrained(output)
tokenizer.save_pretrained(output)
