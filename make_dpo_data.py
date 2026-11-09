"""
Build a DPO preference dataset from the SFT LoRA model.

For each article we sample K candidate summaries from the SFT model, score each
against the reference highlights with ROUGE, and keep (best, worst) as
(chosen, rejected). Pairs whose score gap is too small are dropped.

Output: ./dpo_data/train.jsonl and ./dpo_data/val.jsonl with columns
        prompt / chosen / rejected  (the format TRL's DPOTrainer expects)

Usage:
    python make_dpo_data.py --start 1000 --num 600 --k 4
"""
import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import PeftModel
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_NAME

SFT_PATH = "./peft-text-summary"
OUT_DIR = "./dpo_data"
MAX_ARTICLE_TOKENS = 384  # SFT saw 512 tokens total (article + summary)
MAX_NEW_TOKENS = 128


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device):
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def build_prompt(tokenizer, article):
    """Same prompt template the SFT model was trained with (dataset.py)."""
    ids = tokenizer(article, add_special_tokens=False)["input_ids"][:MAX_ARTICLE_TOKENS]
    article = tokenizer.decode(ids, skip_special_tokens=True)
    return f"Summarize the following text:\n\n{article}\n\n### Summary:\n"


def clean(text):
    # SFT data had no explicit stop marker, so cut at anything that looks like a new section
    for stop in ["\n\n###", "\n\nSummarize the following", "### Summary:"]:
        text = text.split(stop)[0]
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1000, help="first CNN/DM train row (SFT used 0-999)")
    ap.add_argument("--num", type=int, default=600, help="number of articles to sample")
    ap.add_argument("--k", type=int, default=4, help="samples per article")
    ap.add_argument("--batch", type=int, default=2, help="articles per generation batch")
    ap.add_argument("--min_margin", type=float, default=0.05, help="min ROUGE gap chosen-rejected")
    ap.add_argument("--val_frac", type=float, default=0.1)
    args = ap.parse_args()

    device = get_device()
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(SFT_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched generation

    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=get_dtype(device)).to(device)
    model = PeftModel.from_pretrained(base, SFT_PATH).merge_and_unload()
    model.eval()

    data = load_dataset("cnn_dailymail", "3.0.0", split="train")
    data = data.select(range(args.start, args.start + args.num))

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def score(pred, ref):
        if not pred:
            return 0.0
        s = scorer.score(ref, pred)
        return (s["rouge1"].fmeasure + s["rouge2"].fmeasure + s["rougeL"].fmeasure) / 3

    pairs, skipped = [], 0
    for i in tqdm(range(0, len(data), args.batch)):
        batch = data[i : i + args.batch]
        prompts = [build_prompt(tokenizer, a) for a in batch["article"]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                num_return_sequences=args.k,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)

        for j, (prompt, ref) in enumerate(zip(prompts, batch["highlights"])):
            cands = [clean(t) for t in texts[j * args.k : (j + 1) * args.k]]
            cands = list({c: None for c in cands if c})  # dedupe, drop empty
            if len(cands) < 2:
                skipped += 1
                continue
            scored = sorted(((score(c, ref), c) for c in cands), reverse=True)
            (best_s, best), (worst_s, worst) = scored[0], scored[-1]
            if best_s - worst_s < args.min_margin:
                skipped += 1
                continue
            pairs.append({
                "prompt": prompt,
                "chosen": best,
                "rejected": worst,
                "score_chosen": best_s,
                "score_rejected": worst_s,
            })

    print(f"kept {len(pairs)} pairs, skipped {skipped}")
    os.makedirs(OUT_DIR, exist_ok=True)
    n_val = max(1, int(len(pairs) * args.val_frac))
    for name, rows in [("val", pairs[:n_val]), ("train", pairs[n_val:])]:
        with open(os.path.join(OUT_DIR, f"{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows)} rows to {OUT_DIR}/{name}.jsonl")


if __name__ == "__main__":
    main()
