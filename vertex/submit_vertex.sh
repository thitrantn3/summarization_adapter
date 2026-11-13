#!/usr/bin/env bash
# Submit DPO training as a Vertex AI custom job.
#
# The job clones this repo (same as dpo_colab.ipynb), downloads the DPO data,
# trains, and writes the adapter to gs://$BUCKET/dpo-runs/<run>/dpo-text-summary.
# Code must be pushed to $BRANCH first: the job runs what's on GitHub, not local files.
#
#   PROJECT=my-project BUCKET=my-bucket ./vertex/submit_vertex.sh
#   GPU=a100 NUM=3000 PROJECT=... BUCKET=... ./vertex/submit_vertex.sh
#
# Evaluate a finished run (base vs SFT vs DPO with evaluate_dpo.py); results go to
# gs://$BUCKET/dpo-runs/<ADAPTER_RUN>/eval-<time>/. NUM must match the training run
# so the held-out val split is the same one training never saw.
#   MODE=eval ADAPTER_RUN=dpo-20261112-211044 PROJECT=... BUCKET=... ./vertex/submit_vertex.sh
set -euo pipefail

: "${PROJECT:?set PROJECT (GCP project id)}"
: "${BUCKET:?set BUCKET (GCS bucket name, no gs://)}"
MODE="${MODE:-train}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-https://github.com/thitrantn3/summarization_adapter.git}"
BRANCH="${BRANCH:-dpo-training}"
NUM="${NUM:-5000}"
EPOCHS="${EPOCHS:-1}"
EVAL_NUM="${EVAL_NUM:-200}"
GPU="${GPU:-l4}"
STAMP="$(date +%Y%m%d-%H%M%S)"
case "$MODE" in
  train) RUN="dpo-${STAMP}"
         OUT="/gcs/${BUCKET}/dpo-runs/${RUN}" ;;   # Vertex mounts GCS buckets under /gcs/
  eval)  : "${ADAPTER_RUN:?set ADAPTER_RUN to a finished training run, e.g. dpo-20261112-211044}"
         RUN="eval-${STAMP}"
         ADAPTER="/gcs/${BUCKET}/dpo-runs/${ADAPTER_RUN}/dpo-text-summary"
         OUT="/gcs/${BUCKET}/dpo-runs/${ADAPTER_RUN}/${RUN}"
         gcloud storage ls "gs://${ADAPTER#/gcs/}/adapter_model.safetensors" >/dev/null \
           || { echo "No adapter at gs://${ADAPTER#/gcs/}" >&2; exit 1; } ;;
  *) echo "MODE must be train or eval" >&2; exit 1 ;;
esac

case "$GPU" in
  l4)    # 24 GB, ~$1/h: keep gradient checkpointing, modest batch
         MACHINE=g2-standard-12; ACCEL=NVIDIA_L4
         SPEED_ARGS="--batch_size 2 --grad_accum 4" ;;
  a100)  # 40 GB, ~$3-4/h, ~2-3x faster than L4
         MACHINE=a2-highgpu-1g; ACCEL=NVIDIA_TESLA_A100
         SPEED_ARGS="--no_grad_ckpt --batch_size 4 --grad_accum 2" ;;
  *) echo "GPU must be l4 or a100" >&2; exit 1 ;;
esac

IMAGE="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"

# Runs inside the job container. \$ = expanded at runtime; plain $ = filled in now.
# Telegram (optional): set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID when submitting.
JOB_SCRIPT="set -euo pipefail
exec > >(tee /tmp/job.log) 2>&1
notify() { [ -f /app/vertex/notify.py ] && python3 /app/vertex/notify.py \"\$1\" || true; }
trap 'exit 143' TERM
trap 'rc=\$?; [ \$rc -ne 0 ] && notify \"❌ ${RUN} failed or was cancelled (exit \$rc, ${REGION}). Last log lines:
\$(tail -n 15 /tmp/job.log)\"' EXIT
set -x
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git)
git clone -q -b ${BRANCH} ${REPO} /app && cd /app
notify \"🚀 ${RUN} (${MODE}) started on ${GPU} in ${REGION} (NUM=${NUM})\"
# Image ships torch 2.4; current transformers needs >=2.6. cu124 matches the image's CUDA.
# torchvision/torchaudio must be upgraded in lockstep, or importing transformers breaks.
pip install -q torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
printf 'torch==2.6.0\ntorchvision==0.21.0\ntorchaudio==2.6.0\n' > /tmp/torch-pins.txt
pip install -q -U -c /tmp/torch-pins.txt 'trl>=0.19' peft accelerate datasets rouge-score evaluate
# Fail fast (before the 6 GB model download) if the stack doesn't import together
python -c 'import torch, transformers, peft, trl; from transformers import AutoModelForCausalLM; from trl import DPOTrainer; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"transformers\", transformers.__version__, \"peft\", peft.__version__, \"trl\", trl.__version__)'
pip uninstall -q -y torchao || true
python load_hf_dpo_data.py --num ${NUM} --out_dir dpo_data/doc_sum
"
if [ "$MODE" = train ]; then
JOB_SCRIPT+="python train_dpo.py --epochs ${EPOCHS} ${SPEED_ARGS} ${EXTRA_ARGS:-} \
  --ckpt_dir /tmp/qwen-dpo --output_dir ${OUT}/dpo-text-summary
echo done > ${OUT}/_SUCCESS
notify \"✅ ${RUN} finished. Adapter: gs://${BUCKET}/dpo-runs/${RUN}/dpo-text-summary\""
else
JOB_SCRIPT+="cp -r ${ADAPTER} /app/dpo-text-summary
mkdir -p ${OUT}
# DPO first (the one we care about); each model's CSV is saved as soon as it's done
for m in dpo sft base; do
  python evaluate_dpo.py --model \$m --num ${EVAL_NUM}
  cp results/\$m.csv ${OUT}/
  notify \"📊 ${RUN}: \$m done (\$(date -u +%H:%M) UTC)\"
done
python evaluate_dpo.py --compare | tee results/summary.txt
cp results/summary.csv results/summary.txt ${OUT}/
echo done > ${OUT}/_SUCCESS
notify \"✅ ${RUN} finished (${ADAPTER_RUN}, ${EVAL_NUM} prompts):
\$(cat results/summary.txt)
Results: gs://${OUT#/gcs/}\""
fi

CONFIG="$(mktemp -t vertex-dpo).yaml"
python3 - "$CONFIG" "$MACHINE" "$ACCEL" "$IMAGE" "$JOB_SCRIPT" <<'EOF'
import json, os, sys
path, machine, accel, image, script = sys.argv[1:]
env = [{"name": k, "value": os.environ[k]}
       for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID") if os.environ.get(k)]
spec = {"workerPoolSpecs": [{
    "machineSpec": {"machineType": machine, "acceleratorType": accel, "acceleratorCount": 1},
    "replicaCount": 1,
    "diskSpec": {"bootDiskType": "pd-ssd", "bootDiskSizeGb": 100},
    "containerSpec": {"imageUri": image, "command": ["bash", "-c"], "args": [script], "env": env},
}]}
json.dump(spec, open(path, "w"), indent=2)  # JSON is valid YAML
EOF

echo "Submitting $RUN ($MODE, $GPU, NUM=$NUM) -> gs://${OUT#/gcs/}"
gcloud ai custom-jobs create \
  --project="$PROJECT" --region="$REGION" \
  --display-name="$RUN" --config="$CONFIG"
python3 "$(dirname "$0")/notify.py" "⏳ ${RUN} submitted to ${REGION}, waiting for a ${GPU} GPU"
echo
echo "Logs:     gcloud ai custom-jobs stream-logs <JOB_ID> --project=$PROJECT --region=$REGION"
if [ "$MODE" = train ]; then
  echo "Download: gcloud storage cp -r gs://${OUT#/gcs/}/dpo-text-summary ."
else
  echo "Results:  gcloud storage cp -r gs://${OUT#/gcs/} results/"
fi
