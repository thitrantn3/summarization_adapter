"""
Batch summarization with the DPO model (base + merged SFT adapter + DPO adapter).

    python summarize_dpo.py --input docs.csv --output summaries.csv
    python summarize_dpo.py --input docs.jsonl --output out.csv --model base,sft,dpo   # side by side

Input formats:
  .csv / .jsonl  one document per row in a `text` column. Optional `instruction`
                 column (e.g. "Summarize this earnings report for an investor");
                 any other columns (id, title, ...) are copied to the output.
  .txt           documents separated by a line containing only ---

Output: the input columns plus, per model, <model>_summary, <model>_words,
<model>_number_recall (share of the source's numbers kept) and <model>_bullets.
Several models run one after another on the same GPU and land in the same row.

Prompts use the same format as DPO training:  <instruction>:\n\n'<text>'\n\n### Summary:\n
Needs a GPU in practice (3B model); runs on CPU only very slowly.
"""
import argparse
import csv
import gc
import json
import os
import re
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_NAME

SFT_ADAPTER = "./peft-text-summary"
DPO_ADAPTER = "./dpo-text-summary"
DEFAULT_INSTRUCTION = "Summarize this document"
STOPS = ["\n\n###", "\n\nSummarize the following", "### Summary:"]
MODELS = ["base", "sft", "dpo"]
NUM_RE = re.compile(r"\d[\d,.]*%?")  # same as evaluate_dpo.py


def read_docs(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    elif ext == ".jsonl":
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    elif ext == ".txt":
        with open(path, encoding="utf-8") as f:
            chunks = f.read().split("\n---\n")
        rows = [{"text": c.strip()} for c in chunks if c.strip()]
    else:
        sys.exit(f"unsupported input type {ext!r}: use .csv, .jsonl or .txt")
    if not rows:
        sys.exit(f"no documents found in {path}")
    if "text" not in rows[0]:
        sys.exit(f"{path} needs a 'text' column; found {list(rows[0])}")
    return [r for r in rows if (r.get("text") or "").strip()]


def build_prompt(row):
    instruction = (row.get("instruction") or DEFAULT_INSTRUCTION).strip().rstrip(":")
    return f"{instruction}:\n\n'{row['text'].strip()}'\n\n### Summary:\n"


def load_model(which):
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, device_map="auto")
    if which in ("sft", "dpo"):
        model = PeftModel.from_pretrained(model, SFT_ADAPTER).merge_and_unload()
    if which == "dpo":  # DPO adapter was trained on top of the merged SFT model
        model = PeftModel.from_pretrained(model, DPO_ADAPTER).merge_and_unload()
    model.eval()
    tok = AutoTokenizer.from_pretrained(SFT_ADAPTER)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for batched generation with a decoder-only model
    return model, tok


def clean(text):
    for s in STOPS:  # no reliable stop token after the summary
        text = text.split(s)[0]
    return text.strip()


def numbers(text):
    return {n.rstrip(".,") for n in NUM_RE.findall(text)}


def metrics(source, summary):
    src = numbers(source)
    return {
        "words": len(summary.split()),
        "number_recall": round(len(src & numbers(summary)) / len(src), 3) if src else "",
        "bullets": len(re.findall(r"^\s*(?:[-*•]|\d+\.)\s", summary, re.M)),
    }


def write_csv(path, rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:  # BOM so Excel reads UTF-8
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True, help=".csv")
    ap.add_argument("--model", default="dpo",
                    help="one or more of base,sft,dpo, comma-separated (e.g. base,sft,dpo)")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    a = ap.parse_args()
    models = [m.strip() for m in a.model.split(",") if m.strip()]
    bad = [m for m in models if m not in MODELS]
    if bad or not models:
        sys.exit(f"--model must be from {MODELS}, got {a.model!r}")

    docs = read_docs(a.input)
    print(f"{len(docs)} documents from {a.input}; models: {', '.join(models)}")
    rows = [dict(r) for r in docs]

    for which in models:
        model, tok = load_model(which)
        for start in range(0, len(docs), a.batch_size):
            batch = docs[start:start + a.batch_size]
            enc = tok([build_prompt(r) for r in batch], return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                ids = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            texts = tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, text in enumerate(texts):
                summary = clean(text)
                row = rows[start + i]
                row[f"{which}_summary"] = summary
                row.update({f"{which}_{k}": v for k, v in metrics(row["text"], summary).items()})
            print(f"{which} {min(start + a.batch_size, len(docs))} / {len(docs)}", flush=True)
            # rewrite as we go, so partial results survive a crash (unfinished cells stay empty)
            write_csv(a.output, rows)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"wrote {len(rows)} summaries x {len(models)} models -> {a.output}")


if __name__ == "__main__":
    main()
