from datasets import load_dataset
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH, APITOKEN

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,token=APITOKEN, max_length=512, padding="max_length", truncation=True)
tokenizer.pad_token = tokenizer.eos_token


def load_cnn_dailymail(num_train=1000, num_val=200):
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    train_dataset = dataset["train"].select(range(1000))
    val_dataset = dataset["validation"].select(range(200))

    #CREATE PROMPT
    def make_prompt(example):
        instruction = "Summarize the following text:"
        input_text = example["article"]
        output_text = example["highlights"]

        prompt = f"{instruction}\n\n{input_text}\n\n### Summary:\n{output_text}"
        return {"text": prompt}

    train_dataset = train_dataset.map(make_prompt)
    val_dataset = val_dataset.map(make_prompt)
    return train_dataset, val_dataset

def tokenizer_fn(examples):
    # For causal LM, labels are same as input_ids
    examples = tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)

    return examples

def tokenize_dataset(train_dataset, val_dataset):
    #TOKENIZE DATASET
    tokenized_train = train_dataset.map(tokenizer_fn, batched=True)
    tokenized_val = val_dataset.map(tokenizer_fn, batched=True)

    tokenized_train = tokenized_train.map(lambda x: {"labels": x["input_ids"][:]})
    tokenized_val = tokenized_val.map(lambda x: {"labels": x["input_ids"][:]})

    # SET PYTORCH TENSORS
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask",'labels'])
    tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask",'labels'])

    return tokenized_train, tokenized_val

train_dataset, val_dataset = load_cnn_dailymail(1000,200)
tokenized_train, tokenized_val = tokenize_dataset(load_cnn_dailymail(1000,200)[0],load_cnn_dailymail(1000,200)[1])

# Save in raw_data
train_dataset.to_csv("./r_data/r_train.csv")
val_dataset.to_csv("./r_data/r_val.csv")

# Save in tokenized_data
tokenized_train.save_to_disk("./t_data/tokenized_train")
tokenized_val.save_to_disk("./t_data/tokenized_val")

