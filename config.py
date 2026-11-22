import os

MODEL_NAME = "Qwen/Qwen2.5-3B"  # base model
# --- SFT (dataset.py / train.py) ---
MAX_LENGTH = 1024          # prompt + summary tokens; the article is truncated to fit, never the summary
MAX_SUMMARY_TOKENS = 256   # CNN/DailyMail highlights are ~60-100 tokens; longer ones are cut
NUM_TRAIN = 1000           # CNN/DailyMail examples
NUM_VAL = 200
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRAD_ACCUM = 4             # effective batch = TRAIN_BATCH_SIZE * GRAD_ACCUM
LEARNING_RATE = 1e-4       # what the original SFT run used
NUM_EPOCHS = 3
EVAL_STEPS = 50            # eval + checkpoint every N steps (must match for load_best_model_at_end)
EARLY_STOPPING_PATIENCE = 3
OUTPUT_DIR = "./qwen-lora"            # intermediate checkpoints
SFT_OUTPUT = "./peft-text-summary-v2"  # final adapter (v1 in ./peft-text-summary is what DPO was trained on)
# Hugging Face token from the environment (export HF_TOKEN=...)
APITOKEN = os.environ.get("HF_TOKEN")
