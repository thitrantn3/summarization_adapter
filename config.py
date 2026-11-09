import os

MODEL_NAME = "Qwen/Qwen2.5-3B"  # base model
MAX_LENGTH = 512
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
LEARNING_RATE = 2e-4
MAX_STEPS = 1000
OUTPUT_DIR = "./outputs"
# Hugging Face token from the environment (export HF_TOKEN=... or Colab Secrets). None = anonymous.
APITOKEN = os.environ.get("HF_TOKEN")
