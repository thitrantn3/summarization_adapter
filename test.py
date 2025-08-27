import torch
from transformers import AutoTokenizer
from config import MODEL_NAME
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

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

# Generation function
def summarize_text(model, tokenizer, text, max_input_length=512, max_output_length=128):
    # Prepare prompt
    prompt = f"Summarize the following article:\n\n{text}\n\n### Summary:\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # Generate summary
    with torch.no_grad():
        summary_ids = ft_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_output_length,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )

    pred = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    pred = pred.split("Summary:\n")[1].strip() if "Summary:\n" in pred else ""

    return pred

# Example text
article = """
There is room for improvement in the way bullying in schools is handled, particularly in cases where schools may delay communication with parents because they need time to establish the facts, said Education Minister Desmond Lee on Aug 27.
This could unintentionally make parents and children feel anxious, he said in his first remarks to the media about the bullying issue. Any form of hurtful behaviour is wrong and unacceptable, Mr Lee added, and parents and schools need to work together closely to build trust.
The need for clearer and more timely communication with parents was among the findings from consultations with teachers, as part of a Ministry of Education (MOE) review, which began in 2025 to improve processes to address bullying and hurtful behaviour in schools.
Speaking to the media before a dialogue held at the MOE headquarters in Buona Vista, Mr Lee identified four areas that the ministry will be looking into: strengthening school culture and processes; focusing more on values education for students; supporting educators; and improving schools’ partnerships with parents.
"""

summary = summarize_text(ft_model, tokenizer, article)
print(f"Given article: \n{article} \nGenerated summary: \n{summary}")
