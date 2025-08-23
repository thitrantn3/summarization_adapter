# LOAD MODEL
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from config import MODEL_NAME, APITOKEN
import torch

def get_lora_model():
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,token=APITOKEN)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map={"": device},torch_dtype=torch.float32, token=APITOKEN)

    # #CONFIGURE THE ADAPTOR
    lora_config = LoraConfig(
        r=6,               # rank_dimension
        lora_alpha=8,      # scaling factor
        target_modules=["q_proj","v_proj"],  # modules to apply LoRA (attention)
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Wrap the base model with PEFT's get_peft_model function
    peft_model = get_peft_model(model, lora_config)

    # move all weights to GPU (avoids meta tensor issues)
    peft_model.to(device)

    # Quick check
    for name, p in peft_model.named_parameters():
        if p.device.type == "meta":
            print(f"Meta tensor still exists: {name}")  # Should print nothing
    return model

