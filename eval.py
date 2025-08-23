import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from peft import PeftModel
from model import get_lora_model
from dataset import load_cnn_dailymail_dataset
from rouge_score import rouge_scorer

# Load tokenizer
MODEL_NAME = "Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load LoRA model
base_model = get_lora_model()  # returns base + LoRA applied
base_model.eval()

# Load validation dataset
val_dataset = load_cnn_dailymail_dataset(split="validation")

# Evaluation function
def evaluate_model(model, dataset, tokenizer):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

    for sample in dataset:
        input_ids = sample['input_ids'].unsqueeze(0).to(model.device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(model.device)

        # Generate summary
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False
            )

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ref = sample['highlights']

        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    print(f"ROUGE-1: {sum(rouge1_scores)/len(rouge1_scores):.4f}")
    print(f"ROUGE-2: {sum(rouge2_scores)/len(rouge2_scores):.4f}")
    print(f"ROUGE-L: {sum(rougeL_scores)/len(rougeL_scores):.4f}")

# Run evaluation
evaluate_model(base_model, val_dataset, tokenizer)
