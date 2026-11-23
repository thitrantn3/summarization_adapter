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
# SFT_RUN=<sft-run> starts DPO from that MODE=sft run's adapter instead of the repo's original
# ./peft-text-summary. The SFT adapter used is saved next to the DPO adapter (<run>/sft-adapter/),
# and MODE=eval / MODE=infer pick it up automatically. EVAL_AFTER=1 runs the DPO-vs-SFT
# evaluation (evaluate_dpo.py) in the same job once training is done.
#   SFT_RUN=sft-20261122-222719 EVAL_AFTER=1 PROJECT=... BUCKET=... ./vertex/submit_vertex.sh
#
# DATASET / TEMPLATE pick another preference dataset (prompt/chosen/rejected columns) and the
# load_hf_dpo_data.py prompt template (sft | instruct | none). EVAL_AFTER always scores on the
# summarization held-out set (NUM=5000 split), so runs on any dataset stay comparable.
#   DATASET=HumanLLMs/Human-Like-DPO-Dataset TEMPLATE=instruct SFT_RUN=... EVAL_AFTER=1 ...
#
# Evaluate a finished run (base vs SFT vs DPO with evaluate_dpo.py); results go to
# gs://$BUCKET/dpo-runs/<ADAPTER_RUN>/eval-<time>/. NUM must match the training run
# so the held-out val split is the same one training never saw.
#   MODE=eval ADAPTER_RUN=dpo-20261112-211044 PROJECT=... BUCKET=... ./vertex/submit_vertex.sh
#
# Summarize your own documents with a trained adapter (summarize_dpo.py). INPUT is one or
# more comma-separated local .csv/.jsonl (`text` column, optional `instruction`) or .txt
# (docs split by ---) files; each is uploaded and gets gs://$BUCKET/inference/<run>/summaries-<name>.csv.
# INFER_MODEL picks the model(s), side by side in the same CSV (default dpo).
#   MODE=infer INPUT=docs.csv ADAPTER_RUN=dpo-20261112-211044 PROJECT=... BUCKET=... ./vertex/submit_vertex.sh
#   MODE=infer INPUT=a.jsonl,b.jsonl INFER_MODEL=base,sft,dpo ADAPTER_RUN=... PROJECT=... BUCKET=... ./vertex/submit_vertex.sh
#
# SFT on CNN/DailyMail (train.py, settings in config.py, data from the repo's t_data/ and
# r_data/), then eval.py on the new adapter and on the original one for comparison.
# Adapter, eval scores/predictions and the loss log go to gs://$BUCKET/sft-runs/<run>/.
#   MODE=sft PROJECT=... BUCKET=... ./vertex/submit_vertex.sh
#
# Style warm-up before DPO (train.py --jsonl): a short SFT pass on the *chosen* side of local
# DPO data, continuing from the repo's peft-text-summary (or from SFT_RUN's adapter).
# DATA_DIR (default dpo_data/genz) must hold train.jsonl and val.jsonl; dpo_data/ is
# gitignored, so the files are uploaded to gs://$BUCKET/sft-runs/<run>/data/ at submit time.
# WARMUP_EPOCHS / WARMUP_LR override train.py's warm-up defaults (1 epoch, 5e-5).
# The adapter goes to gs://$BUCKET/sft-runs/<run>/peft-genz-sft. Then run DPO on the same data
# from the warmed-up model (DATA_DIR in MODE=train uploads local data instead of DATASET):
#   MODE=warmup PROJECT=... BUCKET=... ./vertex/submit_vertex.sh
#   SFT_RUN=warmup-20261124-101500 DATA_DIR=dpo_data/genz PROJECT=... BUCKET=... ./vertex/submit_vertex.sh
set -euo pipefail

# Telegram credentials can live in ~/.telegram.env (outside the repo) instead of the shell
[ -f "$HOME/.telegram.env" ] && . "$HOME/.telegram.env"
: "${PROJECT:?set PROJECT (GCP project id)}"
: "${BUCKET:?set BUCKET (GCS bucket name, no gs://)}"
MODE="${MODE:-train}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-https://github.com/thitrantn3/summarization_adapter.git}"
BRANCH="${BRANCH:-dpo-training}"
NUM="${NUM:-5000}"
EPOCHS="${EPOCHS:-1}"
EVAL_NUM="${EVAL_NUM:-200}"
DATASET="${DATASET:-stindardlogic/document-summarization-dpo-100k}"
TEMPLATE="${TEMPLATE:-sft}"
GPU="${GPU:-l4}"
STAMP="$(date +%Y%m%d-%H%M%S)"

# Adapter saved by a MODE=sft (peft-text-summary-v2) or MODE=warmup (peft-genz-sft) run
sft_run_adapter() {
  local d order="peft-text-summary-v2 peft-genz-sft"
  case "$1" in warmup-*) order="peft-genz-sft peft-text-summary-v2" ;; esac
  for d in $order; do
    if gcloud storage ls "gs://${BUCKET}/sft-runs/$1/${d}/adapter_model.safetensors" >/dev/null 2>&1; then
      echo "/gcs/${BUCKET}/sft-runs/$1/${d}"; return 0
    fi
  done
  echo "No SFT or warm-up adapter in gs://${BUCKET}/sft-runs/$1/" >&2; return 1
}
# Upload local DATA_DIR/{train,val}.jsonl (dpo_data/ is gitignored) to $OUT/data
upload_data_dir() {
  local f
  for f in train.jsonl val.jsonl; do
    [ -f "${DATA_DIR}/${f}" ] || { echo "Missing ${DATA_DIR}/${f}" >&2; exit 1; }
  done
  gcloud storage cp "${DATA_DIR}/train.jsonl" "${DATA_DIR}/val.jsonl" "gs://${OUT#/gcs/}/data/"
}

case "$MODE" in
  train) RUN="dpo-${STAMP}"
         OUT="/gcs/${BUCKET}/dpo-runs/${RUN}"      # Vertex mounts GCS buckets under /gcs/
         if [ -n "${SFT_RUN:-}" ]; then
           SFT_SRC="$(sft_run_adapter "$SFT_RUN")" || exit 1
         else
           SFT_SRC="peft-text-summary"   # original adapter in the repo
         fi
         if [ -n "${DATA_DIR:-}" ]; then
           upload_data_dir
           DATASET="local:${DATA_DIR}"
         fi ;;
  eval)  : "${ADAPTER_RUN:?set ADAPTER_RUN to a finished training run, e.g. dpo-20261112-211044}"
         RUN="eval-${STAMP}"
         ADAPTER="/gcs/${BUCKET}/dpo-runs/${ADAPTER_RUN}/dpo-text-summary"
         OUT="/gcs/${BUCKET}/dpo-runs/${ADAPTER_RUN}/${RUN}"
         gcloud storage ls "gs://${ADAPTER#/gcs/}/adapter_model.safetensors" >/dev/null \
           || { echo "No adapter at gs://${ADAPTER#/gcs/}" >&2; exit 1; } ;;
  infer) : "${ADAPTER_RUN:?set ADAPTER_RUN to a finished training run, e.g. dpo-20261112-211044}"
         : "${INPUT:?set INPUT to local .csv, .jsonl or .txt file(s) of documents, comma-separated}"
         IFS=, read -r -a INPUTS <<< "$INPUT"
         SEEN=","
         for f in "${INPUTS[@]}"; do
           [ -f "$f" ] || { echo "INPUT file not found: $f" >&2; exit 1; }
           case "$f" in *.csv|*.jsonl|*.txt) ;; *) echo "INPUT must be .csv, .jsonl or .txt: $f" >&2; exit 1 ;; esac
           b="$(basename "$f")"
           case "$SEEN" in *",$b,"*) echo "Two INPUT files share the name $b; rename one" >&2; exit 1 ;; esac
           SEEN+="$b,"
         done
         RUN="infer-${STAMP}"
         ADAPTER="/gcs/${BUCKET}/dpo-runs/${ADAPTER_RUN}/dpo-text-summary"
         OUT="/gcs/${BUCKET}/inference/${RUN}"
         INFER_MODEL="${INFER_MODEL:-dpo}"
         for m in ${INFER_MODEL//,/ }; do
           case "$m" in base|sft|dpo) ;; *) echo "INFER_MODEL must be from base,sft,dpo: $m" >&2; exit 1 ;; esac
         done
         gcloud storage ls "gs://${ADAPTER#/gcs/}/adapter_model.safetensors" >/dev/null \
           || { echo "No adapter at gs://${ADAPTER#/gcs/}" >&2; exit 1; }
         INFER_CMDS=""; OUTPUTS=""
         for f in "${INPUTS[@]}"; do
           b="$(basename "$f")"; stem="${b%.*}"
           gcloud storage cp "$f" "gs://${OUT#/gcs/}/inputs/${b}"
           INFER_CMDS+="python summarize_dpo.py --model ${INFER_MODEL} \$SFT_FLAG --input ${OUT}/inputs/${b} --output ${OUT}/summaries-${stem}.csv
"
           OUTPUTS+="gs://${OUT#/gcs/}/summaries-${stem}.csv
"
         done ;;
  sft)   RUN="sft-${STAMP}"
         OUT="/gcs/${BUCKET}/sft-runs/${RUN}" ;;
  warmup) RUN="warmup-${STAMP}"
         OUT="/gcs/${BUCKET}/sft-runs/${RUN}"
         DATA_DIR="${DATA_DIR:-dpo_data/genz}"
         if [ -n "${SFT_RUN:-}" ]; then
           INIT_SRC="$(sft_run_adapter "$SFT_RUN")" || exit 1
         else
           INIT_SRC="peft-text-summary"   # original adapter in the repo
         fi
         WARMUP_ARGS=""
         [ -n "${WARMUP_EPOCHS:-}" ] && WARMUP_ARGS+=" --epochs ${WARMUP_EPOCHS}"
         [ -n "${WARMUP_LR:-}" ] && WARMUP_ARGS+=" --lr ${WARMUP_LR}"
         upload_data_dir
         DATASET="local:${DATA_DIR}" ;;
  *) echo "MODE must be train, eval, infer, sft or warmup" >&2; exit 1 ;;
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

SFT_PICK="SFT_FLAG=''
if [ -d ${ADAPTER%/dpo-text-summary}/sft-adapter ]; then
  cp -r ${ADAPTER%/dpo-text-summary}/sft-adapter /app/sft-adapter; SFT_FLAG='--sft_path /app/sft-adapter'
fi
echo \"SFT adapter: \${SFT_FLAG:-repo peft-text-summary (v1)}\"
"
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
notify \"🚀 ${RUN} (${MODE}) started on ${GPU} in ${REGION} (NUM=${NUM}, data ${DATASET})\"
# Image ships torch 2.4; current transformers needs >=2.6. cu124 matches the image's CUDA.
# torchvision/torchaudio must be upgraded in lockstep, or importing transformers breaks.
pip install -q torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
printf 'torch==2.6.0\ntorchvision==0.21.0\ntorchaudio==2.6.0\n' > /tmp/torch-pins.txt
pip install -q -U -c /tmp/torch-pins.txt 'trl>=0.19' peft accelerate datasets rouge-score evaluate
# Fail fast (before the 6 GB model download) if the stack doesn't import together
python -c 'import torch, transformers, peft, trl; from transformers import AutoModelForCausalLM; from trl import DPOTrainer; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"transformers\", transformers.__version__, \"peft\", peft.__version__, \"trl\", trl.__version__)'
pip uninstall -q -y torchao || true
"
if [ "$MODE" = train ] && [ -n "${DATA_DIR:-}" ]; then
DPO_FILES="--train_file dpo_data/local/train.jsonl --val_file dpo_data/local/val.jsonl"
JOB_SCRIPT+="mkdir -p dpo_data/local && cp ${OUT}/data/train.jsonl ${OUT}/data/val.jsonl dpo_data/local/
"
elif [ "$MODE" = train ]; then
DPO_FILES=""   # train_dpo.py defaults to dpo_data/doc_sum
JOB_SCRIPT+="python load_hf_dpo_data.py --dataset ${DATASET} --template ${TEMPLATE} --num ${NUM} --out_dir dpo_data/doc_sum
"
elif [ "$MODE" = eval ]; then
JOB_SCRIPT+="python load_hf_dpo_data.py --num ${NUM} --out_dir dpo_data/doc_sum
"
fi
if [ "$MODE" = train ]; then
JOB_SCRIPT+="cp -r ${SFT_SRC} /app/sft-adapter
mkdir -p ${OUT}
cp -r /app/sft-adapter ${OUT}/sft-adapter   # eval/infer must use the same SFT adapter
echo '${SFT_RUN:-repo peft-text-summary (v1)}' > ${OUT}/sft_source.txt
printf 'dataset=%s\ntemplate=%s\nnum=%s\n' '${DATASET}' '${TEMPLATE}' '${NUM}' > ${OUT}/dpo_data_source.txt
python train_dpo.py --epochs ${EPOCHS} ${SPEED_ARGS} ${EXTRA_ARGS:-} --sft_path /app/sft-adapter ${DPO_FILES} \
  --ckpt_dir /tmp/qwen-dpo --output_dir ${OUT}/dpo-text-summary
echo done > ${OUT}/_SUCCESS
notify \"✅ ${RUN} trained on SFT ${SFT_RUN:-v1}. Adapter: gs://${BUCKET}/dpo-runs/${RUN}/dpo-text-summary\"
"
if [ "${EVAL_AFTER:-0}" = 1 ]; then
JOB_SCRIPT+="cp -r ${OUT}/dpo-text-summary /app/dpo-text-summary
mkdir -p ${OUT}/eval
# summarization held-out prompts (same split as the first DPO run), whatever DATASET was trained on
python load_hf_dpo_data.py --num 5000 --out_dir dpo_data/sum_eval
for m in dpo sft; do
  python evaluate_dpo.py --model \$m --num ${EVAL_NUM} --sft_path /app/sft-adapter --data dpo_data/sum_eval/val.jsonl
  cp results/\$m.csv ${OUT}/eval/
  notify \"📊 ${RUN}: \$m evaluated (\$(date -u +%H:%M) UTC)\"
done
python evaluate_dpo.py --compare | tee results/summary.txt
cp results/summary.csv results/summary.txt ${OUT}/eval/
notify \"✅ ${RUN} evaluation on ${EVAL_NUM} held-out summarization prompts (DPO data ${DATASET}, SFT ${SFT_RUN:-v1}):
\$(cat results/summary.txt)
Results: gs://${OUT#/gcs/}/eval\""
fi
fi
if [ "$MODE" = eval ]; then
JOB_SCRIPT+="cp -r ${ADAPTER} /app/dpo-text-summary
${SFT_PICK}mkdir -p ${OUT}
# DPO first (the one we care about); each model's CSV is saved as soon as it's done
for m in dpo sft base; do
  python evaluate_dpo.py --model \$m --num ${EVAL_NUM} \$SFT_FLAG
  cp results/\$m.csv ${OUT}/
  notify \"📊 ${RUN}: \$m done (\$(date -u +%H:%M) UTC)\"
done
python evaluate_dpo.py --compare | tee results/summary.txt
cp results/summary.csv results/summary.txt ${OUT}/
echo done > ${OUT}/_SUCCESS
notify \"✅ ${RUN} finished (${ADAPTER_RUN}, ${EVAL_NUM} prompts):
\$(cat results/summary.txt)
Results: gs://${OUT#/gcs/}\""
elif [ "$MODE" = sft ]; then
JOB_SCRIPT+="pip install -q -c /tmp/torch-pins.txt bert_score nltk absl-py
mkdir -p ${OUT}
python train.py
cp -r peft-text-summary-v2 ${OUT}/
# loss curve: trainer_state.json of the last checkpoint
st=\$(ls -d qwen-lora/checkpoint-* | sort -V | tail -n 1)/trainer_state.json
cp \$st ${OUT}/trainer_state.json
python -c 'import json, sys; s=json.load(open(sys.argv[1])); print(\"best eval_loss\", round(s[\"best_metric\"], 4), \"at\", s[\"best_model_checkpoint\"].split(\"/\")[-1], \"of\", s[\"global_step\"], \"steps\")' \$st | tee /tmp/best.txt
notify \"🏋️ ${RUN}: SFT trained (\$(cat /tmp/best.txt)). Evaluating v2 and v1...\"
python eval.py --model_path ./peft-text-summary-v2 --num ${EVAL_NUM} | tee /tmp/eval_v2.txt
python eval.py --model_path ./peft-text-summary --num ${EVAL_NUM} | tee /tmp/eval_v1.txt
cp -r eval_results ${OUT}/
echo done > ${OUT}/_SUCCESS
notify \"✅ ${RUN} finished. ${EVAL_NUM} CNN/DailyMail val articles:
NEW (v2):
\$(tail -n 4 /tmp/eval_v2.txt)
OLD (v1):
\$(tail -n 4 /tmp/eval_v1.txt)
Adapter: gs://${OUT#/gcs/}/peft-text-summary-v2\""
elif [ "$MODE" = warmup ]; then
JOB_SCRIPT+="mkdir -p dpo_data/warmup && cp ${OUT}/data/train.jsonl ${OUT}/data/val.jsonl dpo_data/warmup/
[ \"${INIT_SRC}\" = peft-text-summary ] || cp -r ${INIT_SRC} /app/init-adapter
INIT=\$([ \"${INIT_SRC}\" = peft-text-summary ] && echo ./peft-text-summary || echo /app/init-adapter)
echo '${SFT_RUN:-repo peft-text-summary (v1)}' > ${OUT}/init_source.txt
python train.py --jsonl dpo_data/warmup/train.jsonl --val_jsonl dpo_data/warmup/val.jsonl \
  --init_adapter \$INIT --output ./peft-genz-sft${WARMUP_ARGS}
cp -r peft-genz-sft ${OUT}/
st=\$(ls -d qwen-lora-warmup/checkpoint-* | sort -V | tail -n 1)/trainer_state.json
cp \$st ${OUT}/trainer_state.json
python -c 'import json, sys; s=json.load(open(sys.argv[1])); print(\"best eval_loss\", round(s[\"best_metric\"], 4), \"at\", s[\"best_model_checkpoint\"].split(\"/\")[-1], \"of\", s[\"global_step\"], \"steps\")' \$st | tee /tmp/best.txt
echo done > ${OUT}/_SUCCESS
notify \"✅ ${RUN} warm-up done (\$(cat /tmp/best.txt)). Adapter: gs://${OUT#/gcs/}/peft-genz-sft
Next: SFT_RUN=${RUN} DATA_DIR=${DATA_DIR} ./vertex/submit_vertex.sh\"
"
elif [ "$MODE" = infer ]; then
JOB_SCRIPT+="cp -r ${ADAPTER} /app/dpo-text-summary
${SFT_PICK}${INFER_CMDS}echo done > ${OUT}/_SUCCESS
notify \"✅ ${RUN} finished (models: ${INFER_MODEL}; adapter ${ADAPTER_RUN}).
\$(grep -o 'wrote [0-9]* summaries x [0-9]* models' /tmp/job.log)
Results:
${OUTPUTS}\""
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

echo "Submitting $RUN ($MODE, $GPU, NUM=$NUM, DATASET=$DATASET, TEMPLATE=$TEMPLATE, SFT_RUN=${SFT_RUN:-v1}) -> gs://${OUT#/gcs/}"
gcloud ai custom-jobs create \
  --project="$PROJECT" --region="$REGION" \
  --display-name="$RUN" --config="$CONFIG"
python3 "$(dirname "$0")/notify.py" "⏳ ${RUN} submitted to ${REGION}, waiting for a ${GPU} GPU"
echo
echo "Logs:     gcloud ai custom-jobs stream-logs <JOB_ID> --project=$PROJECT --region=$REGION"
if [ "$MODE" = train ]; then
  echo "Download: gcloud storage cp -r gs://${OUT#/gcs/}/dpo-text-summary ."
elif [ "$MODE" = eval ]; then
  echo "Results:  gcloud storage cp -r gs://${OUT#/gcs/} results/"
elif [ "$MODE" = sft ]; then
  echo "Results:  gcloud storage cp -r gs://${OUT#/gcs/} results/"
elif [ "$MODE" = warmup ]; then
  echo "Adapter:  gcloud storage cp -r gs://${OUT#/gcs/}/peft-genz-sft ."
  echo "Then DPO: SFT_RUN=${RUN} DATA_DIR=${DATA_DIR} PROJECT=${PROJECT} BUCKET=${BUCKET} ./vertex/submit_vertex.sh"
else
  echo "Results:  gcloud storage cp \"gs://${OUT#/gcs/}/summaries-*.csv\" results/"
fi
