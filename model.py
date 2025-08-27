from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from config import MODEL_NAME, APITOKEN
import torch

def get_lora_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device is {device}')
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map={"": device},torch_dtype=torch.float32, token=APITOKEN)

    # Freeze base model parameters
    for param in base_model.parameters():
        param.requires_grad = False

    # Configure the adaptor
    lora_config = LoraConfig(
        r=64,               # rank_dimension
        lora_alpha=128,      # scaling factor
        target_modules=["q_proj","v_proj"],  # modules to apply LoRA (attention)
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Wrap the base model with PEFT's get_peft_model function
    peft_model = get_peft_model(base_model, lora_config)

    # Move all weights to GPU
    peft_model.to(device)

    # Check if the model's parameters are still "meta" tensors
    for name, p in peft_model.named_parameters():
        if p.device.type == "meta":
            print(f"Meta tensor still exists: {name}")  # Should print nothing

    return peft_model

#Print out the model parameters to make sure the base model params are frozen and only lora params are True
# peft_model = get_lora_model()
# for name, param in peft_model.named_parameters():
#     print(name, param.requires_grad)