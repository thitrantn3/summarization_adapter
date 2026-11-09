#!/usr/bin/env bash
# Submit DPO training as a Vertex AI custom job.
#
# The job clones this repo (same as dpo_colab.ipynb), downloads the DPO data,
# trains, and writes the adapter to gs://$BUCKET/dpo-runs/<run>/dpo-text-summary.
# Code must be pushed to $BRANCH first: the job runs what's on GitHub, not local files.
#
#   PROJECT=my-project BUCKET=my-bucket ./vertex/submit_vertex.sh
#   GPU=a100 NUM=3000 PROJECT=... BUCKET=... ./vertex/submit_vertex.sh
set -euo pipefail

: "${PROJECT:?set PROJECT (GCP project id)}"
: "${BUCKET:?set BUCKET (GCS bucket name, no gs://)}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-https://github.com/thitrantn3/summarization_adapter.git}"
BRANCH="${BRANCH:-dpo-training}"
NUM="${NUM:-5000}"
EPOCHS="${EPOCHS:-1}"
GPU="${GPU:-l4}"
RUN="dpo-$(date +%Y%m%d-%H%M%S)"
OUT="/gcs/${BUCKET}/dpo-runs/${RUN}"   # Vertex mounts GCS buckets under /gcs/

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

JOB_SCRIPT="set -euxo pipefail
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git)
git clone -q -b ${BRANCH} ${REPO} /app && cd /app
pip install -q -U 'trl>=0.19' peft accelerate datasets rouge-score evaluate
pip uninstall -q -y torchao || true
python load_hf_dpo_data.py --num ${NUM} --out_dir dpo_data/doc_sum
python train_dpo.py --epochs ${EPOCHS} ${SPEED_ARGS} ${EXTRA_ARGS:-} \
  --ckpt_dir /tmp/qwen-dpo --output_dir ${OUT}/dpo-text-summary
echo done > ${OUT}/_SUCCESS"

CONFIG="$(mktemp -t vertex-dpo).yaml"
python3 - "$CONFIG" "$MACHINE" "$ACCEL" "$IMAGE" "$JOB_SCRIPT" <<'EOF'
import json, sys
path, machine, accel, image, script = sys.argv[1:]
spec = {"workerPoolSpecs": [{
    "machineSpec": {"machineType": machine, "acceleratorType": accel, "acceleratorCount": 1},
    "replicaCount": 1,
    "diskSpec": {"bootDiskType": "pd-ssd", "bootDiskSizeGb": 100},
    "containerSpec": {"imageUri": image, "command": ["bash", "-c"], "args": [script]},
}]}
json.dump(spec, open(path, "w"), indent=2)  # JSON is valid YAML
EOF

echo "Submitting $RUN ($GPU, NUM=$NUM) -> gs://${BUCKET}/dpo-runs/${RUN}"
gcloud ai custom-jobs create \
  --project="$PROJECT" --region="$REGION" \
  --display-name="$RUN" --config="$CONFIG"

echo
echo "Logs:     gcloud ai custom-jobs stream-logs <JOB_ID> --project=$PROJECT --region=$REGION"
echo "Download: gcloud storage cp -r gs://${BUCKET}/dpo-runs/${RUN}/dpo-text-summary ."
