import torch
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from peft import PeftModel
from config import MODEL_NAME
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate import load
import evaluate
import numpy as np
import pandas as pd

# Load validation dataset
tokenized_val = load_from_disk("./t_data/tokenized_val")
r_val = load_from_disk("./r_data/r_val")
print(f'Loaded eval dataset length: {len(tokenized_val)}, {len(r_val)}')

# Load trained model
MODEL_PATH = "./peft-text-summary"  # path where you saved your PEFT model

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load base model (same as the one you trained on)
base_model_name = MODEL_NAME  # or your original base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map={"": device},        # automatically put layers on GPU
    torch_dtype="auto"        # or torch.float16
)

# Load PEFT / LoRA adapters
# ft_model = PeftModel.from_pretrained(model, MODEL_PATH)
model.eval()

# Evaluation function
def evaluate_model(model, t_dataset, r_dataset, tokenizer):

    res = []

    for i in range(len(t_dataset)):
        # print(f"Prompt: {r_val[i]['text']}")
        # print(f"Human-generated summary: {r_val[i]['text'].split('Response:\n')[1]}")
        t_sample = tokenized_val[i]
        input_ids = t_sample['input_ids'].unsqueeze(0).to(model.device)
        attention_mask = t_sample['attention_mask'].unsqueeze(0).to(model.device)

        # Generate summary
        print(f"Generating summary {i}....")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False
            )

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred = pred.split("Response:\n")[1].strip() if "Response:\n" in pred else ""
        print(f"PREDICTION: {len(pred)}")

        r_sample = r_val[i]
        ref = r_sample.get('highlights')
        # print(ref)
        print(f'ref length: {len(ref)}')

        if len(pred) == 0 or len(ref) == 0:
          print("No predictions or references to compute ROUGE.")    
          res.append({
              'rouge1': np.float64(0.0),
              'rouge2': np.float64(0.0),
              'rougeL': np.float64(0.0),
              'rougeLsum': np.float64(0.0)
          })
          continue

        print("Calculating rouge scores...")
        rouge = evaluate.load('rouge')
        results = rouge.compute(predictions=pred,
                            references=ref[:len(pred)],
                            tokenizer=lambda x: x.split(),
                            use_aggregator=True,
                            use_stemmer=True)
        res.append(results)

        return res

# Run evaluation
results = evaluate_model(model, tokenized_val, r_val, tokenizer)

df = pd.DataFrame(results)
print(df)
df.to_csv("./rouge_scores.csv")