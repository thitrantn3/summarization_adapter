"""
Prepare CNN/DailyMail for SFT.

    python dataset.py                       # sizes from config.py
    python dataset.py --num_train 2000 --num_val 200

Each example is  PROMPT + SUMMARY + <eos>  where
    PROMPT = "Summarize the following text:\n\n{article}\n\n### Summary:\n"
- The article is truncated (by tokens) so the whole example fits MAX_LENGTH;
  the summary is never cut (except beyond MAX_SUMMARY_TOKENS).
- labels are -100 on the prompt, so the loss is only on the summary + <eos>.
  The <eos> teaches the model to stop after the summary.
- No padding here; train.py pads each batch dynamically.

Outputs:
    r_data/r_train(.csv), r_data/r_val(.csv)    raw article / highlights / prompt
    t_data/tokenized_train, t_data/tokenized_val  input_ids / attention_mask / labels
"""
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

from config import APITOKEN, MAX_LENGTH, MAX_SUMMARY_TOKENS, MODEL_NAME, NUM_TRAIN, NUM_VAL

INSTRUCTION = "Summarize the following text:"


def get_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=APITOKEN)
    tok.pad_token = tok.eos_token
    return tok


def build_prompt(article, tok, max_prompt_tokens):
    """Prompt with the article truncated so the whole prompt fits max_prompt_tokens."""
    overhead = len(tok(f"{INSTRUCTION}\n\n\n\n### Summary:\n")["input_ids"])
    ids = tok(article.strip())["input_ids"][: max(0, max_prompt_tokens - overhead - 2)]
    article = tok.decode(ids, skip_special_tokens=True)
    return f"{INSTRUCTION}\n\n{article}\n\n### Summary:\n"


def encode(example, tok, max_length=MAX_LENGTH, max_summary_tokens=MAX_SUMMARY_TOKENS):
    summary_ids = tok(example["highlights"].strip())["input_ids"][:max_summary_tokens] + [tok.eos_token_id]
    prompt = build_prompt(example["article"], tok, max_length - len(summary_ids))
    prompt_ids = tok(prompt)["input_ids"]
    input_ids = (prompt_ids + summary_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + summary_ids)[:max_length]
    return {"prompt": prompt, "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids), "labels": labels}


def load_cnn_dailymail(num_train=NUM_TRAIN, num_val=NUM_VAL):
    ds = load_dataset("cnn_dailymail", "3.0.0")
    return ds["train"].select(range(num_train)), ds["validation"].select(range(num_val))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_train", type=int, default=NUM_TRAIN)
    ap.add_argument("--num_val", type=int, default=NUM_VAL)
    ap.add_argument("--max_length", type=int, default=MAX_LENGTH)
    a = ap.parse_args()

    tok = get_tokenizer()
    train, val = load_cnn_dailymail(a.num_train, a.num_val)

    for name, raw in (("train", train), ("val", val)):
        enc = raw.map(lambda x: encode(x, tok, a.max_length), desc=f"tokenize {name}")
        # raw: article / highlights / id / prompt (prompt = what eval.py feeds the model)
        enc.select_columns(["article", "highlights", "id", "prompt"]).to_csv(f"./r_data/r_{name}.csv")
        enc.select_columns(["article", "highlights", "id", "prompt"]).save_to_disk(f"./r_data/r_{name}")
        enc.select_columns(["input_ids", "attention_mask", "labels"]).save_to_disk(f"./t_data/tokenized_{name}")

        lens = sorted(len(x) for x in enc["input_ids"])
        trained = sum(sum(l != -100 for l in x) for x in enc["labels"])
        print(f"{name}: {len(enc)} examples, tokens median {lens[len(lens) // 2]} max {lens[-1]}, "
              f"{trained / len(enc):.0f} label tokens/example on average")


if __name__ == "__main__":
    main()
