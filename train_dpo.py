"""
DPO fine-tuning on top of the SFT LoRA model.

Approach:
  1. Load Qwen2.5-3B + SFT adapter (./peft-text-summary) and merge it -> SFT model.
  2. Attach a fresh LoRA adapter and train it with DPO.
     With ref_model=None, TRL computes reference log-probs by disabling the
     adapter, so the reference is exactly the frozen SFT model (one copy in memory).
  3. Save the DPO adapter to ./dpo-text-summary.

The DPO adapter applies on top of the *merged SFT* model; use
model.load_dpo_model() to load it for inference.

Data: your own preference file (see --help), or the output of make_dpo_data.py.
"""
import argparse
import os

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from config import MODEL_NAME

SFT_PATH = "./peft-text-summary"
DPO_PATH = "./dpo-text-summary"
# Same template the SFT model was trained on (dataset.py)
PROMPT_TEMPLATE = "Summarize the following text:\n\n{article}\n\n### Summary:\n"


def get_device_and_dtype():
    if torch.cuda.is_available():
        bf16 = torch.cuda.is_bf16_supported()
        return torch.device("cuda"), (torch.bfloat16 if bf16 else torch.float16), bf16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32, False
    return torch.device("cpu"), torch.float32, False


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", default="./dpo_data/train.jsonl",
                    help=".jsonl / .json / .csv / .parquet with prompt, chosen, rejected")
    ap.add_argument("--val_file", default=None, help="optional; otherwise split from train")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--prompt_col", default="prompt")
    ap.add_argument("--chosen_col", default="chosen")
    ap.add_argument("--rejected_col", default="rejected")
    ap.add_argument("--wrap_prompt", action="store_true",
                    help="prompt column holds the raw article; wrap it in the SFT template")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    return ap.parse_args()


def _loader(path):
    ext = os.path.splitext(path)[1].lower()
    return {".jsonl": "json", ".json": "json", ".csv": "csv", ".parquet": "parquet"}[ext]


def load_preference_data(cli):
    files = {"train": cli.train_file}
    if cli.val_file:
        files["val"] = cli.val_file
    data = load_dataset(_loader(cli.train_file), data_files=files)

    rename = {cli.prompt_col: "prompt", cli.chosen_col: "chosen", cli.rejected_col: "rejected"}
    rename = {k: v for k, v in rename.items() if k != v}
    if rename:
        data = data.rename_columns(rename)
    data = data.select_columns(["prompt", "chosen", "rejected"])

    if cli.wrap_prompt:
        data = data.map(lambda x: {"prompt": PROMPT_TEMPLATE.format(article=x["prompt"])})

    if "val" not in data:
        split = data["train"].train_test_split(test_size=cli.val_frac, seed=42)
        data = DatasetDict(train=split["train"], val=split["test"])
    return data


def main():
    cli = parse_args()
    device, dtype, use_bf16 = get_device_and_dtype()
    print(f"device: {device}, dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(SFT_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base + SFT adapter and merging...")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base, SFT_PATH).merge_and_unload()
    model.config.use_cache = False

    data = load_preference_data(cli)
    print(f"train pairs: {len(data['train'])}, val pairs: {len(data['val'])}")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    args = DPOConfig(
        output_dir="./qwen-dpo",
        beta=cli.beta,                     # strength of KL pull toward SFT model
        learning_rate=cli.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=cli.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        max_length=1024,                # doc-sum DPO: prompt ~300 + chosen ~300 tokens
        bf16=use_bf16,
        fp16=(device.type == "cuda" and not use_bf16),  # e.g. Colab T4
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,                 # reference = model with adapter disabled = SFT model
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Training DPO...")
    trainer.train()

    print(f"Saving DPO adapter to {DPO_PATH}")
    trainer.model.save_pretrained(DPO_PATH)
    tokenizer.save_pretrained(DPO_PATH)


if __name__ == "__main__":
    main()
