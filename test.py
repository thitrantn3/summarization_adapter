import torch
from transformers import AutoTokenizer
from model import get_lora_model
from config import MODEL_NAME

# 1️⃣ Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = get_lora_model()
model.eval()

# 2️⃣ Generation function
def summarize_text(model, tokenizer, text, max_input_length=512, max_output_length=128):
    # Prepare prompt
    prompt = f"Summarize the following article:\n\n{text}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # Generate summary
    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_output_length,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 3️⃣ Example usage
article = """
The United Nations warned that climate change could displace millions of people over the next decade.
Governments need to take urgent action to reduce carbon emissions to avoid catastrophic effects.
"""

summary = summarize_text(model, tokenizer, article)
print("Generated summary:\n", summary)
