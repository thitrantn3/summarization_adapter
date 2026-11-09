"""
Download a DPO preference dataset from the Hugging Face Hub and save it as
train/val .jsonl files that train_dpo.py can read.

Default: stindardlogic/document-summarization-dpo-100k
  (columns: prompt, chosen, rejected, metadata, id; 100k rows).
  Prompts already contain the instruction ("Summarize this ... document:\n\n'...'"),
  so the default "sft" template just appends the SFT model's "### Summary:" cue.

Usage:
    python load_hf_dpo_data.py
    python load_hf_dpo_data.py --num 5000 --template none
    python train_dpo.py --train_file dpo_data/doc_sum/train.jsonl \
                        --val_file   dpo_data/doc_sum/val.jsonl
"""
import argparse
import os

from datasets import load_dataset

TEMPLATES = {
    # Prompt already holds the instruction; add the cue the SFT model was trained on (dataset.py)
    "sft": "{prompt}\n\n### Summary:\n",
    # Generic instruction wrapper
    "instruct": "### Instruction:\n{prompt}\n\n### Response:\n",
    # Leave prompts untouched
    "none": "{prompt}",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="stindardlogic/document-summarization-dpo-100k")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out_dir", default="./dpo_data/doc_sum")
    ap.add_argument("--num", type=int, default=None, help="use only N rows (after shuffle); 100k is a lot for a 3B model")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--template", choices=TEMPLATES, default="sft")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading {args.dataset} [{args.split}] ...")
    ds = load_dataset(args.dataset, split=args.split)
    print(f"loaded {len(ds)} rows, columns: {ds.column_names}")

    missing = {"prompt", "chosen", "rejected"} - set(ds.column_names)
    if missing:
        raise ValueError(f"dataset is missing columns {missing}; rename them first")
    ds = ds.select_columns(["prompt", "chosen", "rejected"])

    # Drop empty or identical pairs
    before = len(ds)
    ds = ds.filter(lambda x: all(x[c] and x[c].strip() for c in ("prompt", "chosen", "rejected"))
                   and x["chosen"].strip() != x["rejected"].strip())
    print(f"dropped {before - len(ds)} empty/identical rows")

    ds = ds.shuffle(seed=args.seed)
    if args.num:
        ds = ds.select(range(min(args.num, len(ds))))

    tmpl = TEMPLATES[args.template]
    ds = ds.map(lambda x: {"prompt": tmpl.format(prompt=x["prompt"].strip()),
                           "chosen": x["chosen"].strip(),
                           "rejected": x["rejected"].strip()})

    split = ds.train_test_split(test_size=args.val_frac, seed=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, part in [("train", split["train"]), ("val", split["test"])]:
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        part.to_json(path, lines=True, force_ascii=False)
        print(f"wrote {len(part)} rows -> {path}")

    ex = split["train"][0]
    print("\nExample:\n--- prompt ---\n" + ex["prompt"] +
          "\n--- chosen ---\n" + ex["chosen"][:300] +
          "\n--- rejected ---\n" + ex["rejected"][:300])


if __name__ == "__main__":
    main()
