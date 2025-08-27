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
MODEL_PATH = "./peft-text-summary"  

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load base model
base_model_name = MODEL_NAME
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map={"": device},       
    torch_dtype="auto"
)

# Load PEFT / LoRA adapters
ft_model = PeftModel.from_pretrained(base_model, MODEL_PATH)
ft_model.print_trainable_parameters()
ft_model.eval()

def compute_rouge(pred,ref):
    #COMPUTING ROUGE SCORE
    if len(pred) == 0 or len(ref) == 0:
      print("No predictions or references to compute ROUGE.")
      return ({
          'rouge1': np.float64(0.0),
          'rouge2': np.float64(0.0),
          'rougeL': np.float64(0.0),
          'rougeLsum': np.float64(0.0)
      })

    print("Calculating rouge scores...")
    rouge = evaluate.load('rouge')
    r_results = rouge.compute(predictions=[pred],
                        references=[ref],
                        tokenizer=lambda x: x.split(),
                        use_aggregator=True,
                        use_stemmer=True)
    return r_results

def compute_bleu(pred,ref):

    if len(pred) == 0 or len(ref) == 0:
      print("No predictions or references to compute BLEU.")
      return ({'accuracy': np.float64(0.0)})
    bleu_metric = evaluate.load("bleu")
    print("Calculating BLEU scores...")
    bl_results = bleu_metric.compute(predictions=[pred], references=[ref])

    return bl_results

def compute_bert(pred,ref):

    if len(pred) == 0 or len(ref) == 0:
      print("No predictions or references to compute BERT.")
      return ({'accuracy': np.float64(0.0)})
    bertscore_metric = evaluate.load("bertscore")
    print("Calculating BERT scores...")
    be_results = bertscore_metric.compute(predictions=[pred], references=[ref],lang="en")

    return be_results

# Evaluation function
def evaluate_model(model, t_dataset, r_dataset, tokenizer):
    print(f'Evaluating for model {model}')
    r_res = []
    bl_res = []
    be_res = []
    predictions = []
    print(len(r_dataset))

    for i in range(len(r_dataset)):
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
                do_sample=False,
                temperature=0.1
            )
        print(outputs[0])
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred = pred.split("Summary:\n")[1].strip() if "Summary:\n" in pred else ""
        print(f"PREDICTION: {len(pred)}")

        predictions.append(pred)

        r_sample = r_val[i]
        ref = r_sample.get('highlights')
        print(f'ref length: {len(ref)}')

        #Compute rouge
        r_scores = compute_rouge(pred,ref)
        r_res.append(r_scores)

        #Compute bleu
        bl_scores = compute_bleu(pred,ref)
        bl_res.append(bl_scores)

        #Compute bert
        be_scores = compute_bert(pred,ref)
        be_res.append(be_scores)

    return predictions, r_res , bl_res, be_res

# Run evaluation
predictions, r_results, bl_results, be_results = evaluate_model(ft_model, tokenized_val, r_val, tokenizer)

# Save rouge_score/predictions
rouge_df = pd.DataFrame(r_results)
print(f'Saving predictions of length {len(rouge_df)}')
rouge_df.to_csv('./eval_results/rouge_score/rouge_peft.csv')

# Save rouge_score/predictions
bleu_df = pd.DataFrame(bl_results)
print(f'Saving predictions of length {len(bleu_df)}')
bleu_df.to_csv('./eval_results/rouge_score/bleu_peft.csv')

# Save bert_score/predictions
bert_df = pd.DataFrame(be_results)
print(f'Saving predictions of length {len(bert_df)}')
bert_df.to_csv('./eval_results/rouge_score/bert_peft.csv')

# Save the predicted summary
pred_df = pd.DataFrame()
pred_df['prompt'] = r_val['text']
pred_df['ground_truth'] = r_val['highlights']
pred_df['prediction'] = predictions
print(f'Saving predictions of length {len(pred_df)}')
pred_df.to_csv('./eval_results/sum_predictions/pred_peft.csv')