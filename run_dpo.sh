#!/usr/bin/env bash
# End-to-end DPO training on top of the SFT LoRA model (./peft-text-summary).
#
#   ./run_dpo.sh                 # 10k pairs, default hyperparameters
#   NUM=5000 EPOCHS=2 LR=1e-5 BETA=0.1 ./run_dpo.sh
#   NUM=all ./run_dpo.sh         # use the full 100k dataset
#
# Output: ./dpo-text-summary  (DPO LoRA adapter; load with model.load_dpo_model())
set -euo pipefail
cd "$(dirname "$0")"

DATASET="${DATASET:-stindardlogic/document-summarization-dpo-100k}"
DATA_DIR="${DATA_DIR:-./dpo_data/doc_sum}"
NUM="${NUM:-10000}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-5e-6}"
BETA="${BETA:-0.1}"

echo "==> [1/3] Checking dependencies"
python -c "import trl" 2>/dev/null || pip install -r requirements.txt

if [ ! -f peft-text-summary/adapter_model.safetensors ]; then
  echo "SFT adapter not found at ./peft-text-summary — run train.py first." >&2
  exit 1
fi

echo "==> [2/3] Preparing DPO data from $DATASET"
if [ -f "$DATA_DIR/train.jsonl" ] && [ -f "$DATA_DIR/val.jsonl" ]; then
  echo "    found existing $DATA_DIR, skipping download (delete it to rebuild)"
else
  NUM_ARG=()
  [ "$NUM" != "all" ] && NUM_ARG=(--num "$NUM")
  python load_hf_dpo_data.py --dataset "$DATASET" --out_dir "$DATA_DIR" "${NUM_ARG[@]+"${NUM_ARG[@]}"}"
fi

echo "==> [3/3] Training DPO (epochs=$EPOCHS lr=$LR beta=$BETA)"
python train_dpo.py \
  --train_file "$DATA_DIR/train.jsonl" \
  --val_file "$DATA_DIR/val.jsonl" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --beta "$BETA"

echo "==> Done. DPO adapter saved to ./dpo-text-summary"
