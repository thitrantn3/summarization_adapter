# LOAD MODEL
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from config import MODEL_NAME, APITOKEN
import torch

def get_lora_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,token=token)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto",torch_dtype="auto", token=token)

    # #CONFIGURE THE ADAPTOR
    lora_config = LoraConfig(
        r=6,               # rank_dimension
        lora_alpha=8,      # scaling factor
        target_modules=["q_proj","v_proj"],  # modules to apply LoRA (attention)
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    return model

