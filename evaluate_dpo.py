"""
Compare base / SFT / DPO summarization models on held-out DPO prompts.

    python evaluate_dpo.py --model base
    python evaluate_dpo.py --model sft
    python evaluate_dpo.py --model dpo
    python evaluate_dpo.py --compare          # summary table from results/*.csv

Test set: dpo_data/doc_sum/val.jsonl (prompt / chosen / rejected), produced by
load_hf_dpo_data.py. Metrics per response:
  - ROUGE-1/2/L F1 vs the preferred ("chosen") summary
  - number_recall: share of numbers in the source document kept in the summary
    (the dataset's main reason for rejecting summaries is dropping figures)
  - length and formatting: words, chars, bullets, bold, table rows
"""
import argparse, json, re
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_NAME

SFT_ADAPTER = './peft-text-summary'
DPO_ADAPTER = './dpo-text-summary'
MODELS = ['base', 'sft', 'dpo']
STOPS = ['\n\n###', '\n\nSummarize the following', '### Summary:']
NUM_RE = re.compile(r'\d[\d,.]*%?')


def load_model(which):
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, device_map='auto')
    if which in ('sft', 'dpo'):
        model = PeftModel.from_pretrained(model, SFT_ADAPTER)
        if which == 'dpo':  # DPO adapter was trained on top of the merged SFT model
            model = PeftModel.from_pretrained(model.merge_and_unload(), DPO_ADAPTER)
    model.eval()
    tok = AutoTokenizer.from_pretrained(SFT_ADAPTER)
    tok.pad_token = tok.eos_token
    return model, tok


def generate(model, tok, prompt, max_new_tokens):
    inputs = tok(prompt, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    for s in STOPS:  # base/SFT models have no reliable stop token after the summary
        text = text.split(s)[0]
    return text.strip()


def numbers(text):
    return {n.rstrip('.,') for n in NUM_RE.findall(text)}


def score_row(row, output, scorer):
    r = scorer.score(row['chosen'], output) if output else None
    doc = row['prompt'].split('### Summary:')[0]
    src_nums, out_nums = numbers(doc), numbers(output)
    return {
        'rouge1': r['rouge1'].fmeasure if r else 0.0,
        'rouge2': r['rouge2'].fmeasure if r else 0.0,
        'rougeL': r['rougeL'].fmeasure if r else 0.0,
        'number_recall': len(src_nums & out_nums) / len(src_nums) if src_nums else None,
        'words': len(output.split()),
        'chars': len(output),
        'bullets': len(re.findall(r'^\s*(?:[-*•]|\d+\.)\s', output, re.M)),
        'bold': output.count('**') // 2,
        'table_rows': len(re.findall(r'^\s*\|.*\|\s*$', output, re.M)),
        'empty': int(not output),
    }


METRICS = ['rouge1', 'rouge2', 'rougeL', 'number_recall', 'words', 'chars',
           'bullets', 'bold', 'table_rows', 'empty']


def compare():
    frames = [pd.read_csv(f'results/{m}.csv') for m in MODELS if Path(f'results/{m}.csv').exists()]
    if not frames:
        raise SystemExit('no results/*.csv yet — run with --model first')
    df = pd.concat(frames)
    table = df.groupby('model', sort=False)[METRICS].mean().round(3)
    print(table.to_string())
    table.to_csv('results/summary.csv')


p = argparse.ArgumentParser()
p.add_argument('--model', choices=MODELS)
p.add_argument('--compare', action='store_true')
p.add_argument('--data', default='dpo_data/doc_sum/val.jsonl')
p.add_argument('--num', type=int, default=200, help='number of test prompts (0 = all)')
p.add_argument('--max_new_tokens', type=int, default=512)
a = p.parse_args()

if a.compare:
    compare()
elif not a.model:
    p.error('pass --model {base,sft,dpo} or --compare')
else:
    rows = [json.loads(x) for x in open(a.data)]
    if a.num:
        rows = rows[:a.num]
    model, tok = load_model(a.model)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = []
    for i, row in enumerate(rows):
        output = generate(model, tok, row['prompt'], a.max_new_tokens)
        results.append({'id': i, 'model': a.model, 'prompt': row['prompt'],
                        'reference': row['chosen'], 'response': output,
                        **score_row(row, output, scorer)})
        if (i + 1) % 25 == 0:
            print(a.model, i + 1, '/', len(rows))
    Path('results').mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(f'results/{a.model}.csv', index=False)
    print(pd.DataFrame(results)[METRICS].mean().round(3))
