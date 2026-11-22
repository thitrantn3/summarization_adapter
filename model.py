from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from config import MODEL_NAME, APITOKEN
import torch

def train_dtype():
    """bf16 on Ampere+ (A100/L4), fp16 on older GPUs (T4), fp32 on CPU.
    Only the frozen base weights use it; PEFT keeps the LoRA weights in fp32."""
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return torch.float32


def get_lora_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = train_dtype()
    print(f'device is {device}, base dtype {dtype}')
    # fp32 (12 GB for 3B params) doesn't fit next to 1024-token activations on a 16-24 GB GPU
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map={"": device}, torch_dtype=dtype, token=APITOKEN)

    # Freeze base model parameters
    for param in base_model.parameters():
        param.requires_grad = False
    # Gradient checkpointing with a frozen base needs inputs that require grad
    base_model.enable_input_require_grads()

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

def load_dpo_model(sft_path="./peft-text-summary", dpo_path="./dpo-text-summary"):
    """Base model + merged SFT adapter + DPO adapter (for inference/eval)."""
    from peft import PeftModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", token=APITOKEN).to(device)
    sft = PeftModel.from_pretrained(base, sft_path).merge_and_unload()
    model = PeftModel.from_pretrained(sft, dpo_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(dpo_path)
    return model, tokenizer
