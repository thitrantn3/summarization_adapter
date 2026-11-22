"""
Evaluate an SFT adapter on the CNN/DailyMail validation set (run dataset.py first).

    python eval.py                                   # adapter from config.SFT_OUTPUT
    python eval.py --model_path ./peft-text-summary  # the original (v1) adapter

The model sees only the prompt (article + "### Summary:"), never the reference;
only the newly generated tokens are scored against the reference highlights.
Writes eval_results/scores/{rouge,bleu,bert}_<name>.csv and
eval_results/sum_predictions/pred_<name>.csv.
"""
import argparse
import os

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MAX_LENGTH, MODEL_NAME, SFT_OUTPUT
from dataset import build_prompt

STOPS = ["\n\n###", "\n\nSummarize the following", "### Summary:"]

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default=SFT_OUTPUT)
ap.add_argument("--num", type=int, default=0, help="evaluate only the first N examples (0 = all)")
ap.add_argument("--max_new_tokens", type=int, default=128)
a = ap.parse_args()
name = os.path.basename(os.path.normpath(a.model_path)).replace("peft-text-summary", "peft")

# Load validation dataset (article / highlights / prompt)
r_val = load_from_disk("./r_data/r_val")
if "prompt" not in r_val.column_names:
    raise SystemExit("r_data is in the old format (prompt included the answer) - rerun `python dataset.py`")
if a.num:
    r_val = r_val.select(range(min(a.num, len(r_val))))
print(f'Loaded eval dataset length: {len(r_val)}')

tokenizer = AutoTokenizer.from_pretrained(a.model_path)
tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load base model + PEFT / LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map={"": device}, torch_dtype="auto")
ft_model = PeftModel.from_pretrained(base_model, a.model_path)
ft_model.print_trainable_parameters()
ft_model.eval()

rouge = evaluate.load('rouge')
bleu_metric = evaluate.load("bleu")
bertscore_metric = evaluate.load("bertscore")


def compute_rouge(pred, ref):
    if len(pred) == 0 or len(ref) == 0:
        return {'rouge1': np.float64(0.0), 'rouge2': np.float64(0.0),
                'rougeL': np.float64(0.0), 'rougeLsum': np.float64(0.0)}
    return rouge.compute(predictions=[pred], references=[ref], tokenizer=lambda x: x.split(),
                         use_aggregator=True, use_stemmer=True)


def compute_bleu(pred, ref):
    if len(pred) == 0 or len(ref) == 0:
        return {'bleu': np.float64(0.0)}
    return bleu_metric.compute(predictions=[pred], references=[ref])


def compute_bert(pred, ref):
    if len(pred) == 0 or len(ref) == 0:
        return {'precision': [0.0], 'recall': [0.0], 'f1': [0.0]}
    return bertscore_metric.compute(predictions=[pred], references=[ref], lang="en")


def generate(model, article):
    # Shorten the article (not the end of the prompt) so prompt + new tokens fit MAX_LENGTH
    prompt = build_prompt(article, tokenizer, MAX_LENGTH - a.max_new_tokens)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    # decode only the new tokens, not the prompt
    text = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    for s in STOPS:
        text = text.split(s)[0]
    return text.strip()


def evaluate_model(model, r_dataset):
    print(f'Evaluating {a.model_path} on {len(r_dataset)} examples')
    r_res, bl_res, be_res, predictions = [], [], [], []
    for i, row in enumerate(r_dataset):
        pred = generate(model, row['article'])
        ref = row['highlights']
        predictions.append(pred)
        r_res.append(compute_rouge(pred, ref))
        bl_res.append(compute_bleu(pred, ref))
        be_res.append(compute_bert(pred, ref))
        if (i + 1) % 20 == 0:
            print(f"{i + 1} / {len(r_dataset)}")
    return predictions, r_res, bl_res, be_res


predictions, r_results, bl_results, be_results = evaluate_model(ft_model, r_val)

os.makedirs('./eval_results/scores', exist_ok=True)
os.makedirs('./eval_results/sum_predictions', exist_ok=True)

rouge_df = pd.DataFrame(r_results)
rouge_df.to_csv(f'./eval_results/scores/rouge_{name}.csv')
pd.DataFrame(bl_results).to_csv(f'./eval_results/scores/bleu_{name}.csv')
bert_df = pd.DataFrame(be_results)
bert_df.to_csv(f'./eval_results/scores/bert_{name}.csv')

pred_df = pd.DataFrame({'prompt': r_val['prompt'], 'ground_truth': r_val['highlights'],
                        'prediction': predictions})
pred_df.to_csv(f'./eval_results/sum_predictions/pred_{name}.csv')

print(rouge_df[['rouge1', 'rouge2', 'rougeL']].mean().round(3).to_string())
print('bertscore f1', round(float(np.mean([x['f1'][0] for x in be_results])), 3))
