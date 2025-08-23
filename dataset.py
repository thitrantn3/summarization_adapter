from datasets import load_dataset
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH

def load_cnn_dailymail(num_train=1000, num_val=200):
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    train_dataset = dataset["train"].select(range(1000))
    val_dataset = dataset["validation"].select(range(200))

    #CREATE PROMPT
    def make_prompt(example):
        instruction = "Summarize the following article."
        input_text = example["article"]
        output_text = example["highlights"]
        
        prompt = f"{instruction}\n\n{input_text}\n\n### Response:\n{output_text}"
        return {"text": prompt}

    train_dataset = train_dataset.map(make_prompt)
    val_dataset = val_dataset.map(make_prompt)

    #TOKENIZE DATASET
    model_name = "facebook/opt-1.3b"  # example open-weight model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # important for causal LM
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=512,  # adjust based on your GPU
            padding="max_length"
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)

    # SET PYTORCH TENSORS
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return train_dataset, val_dataset