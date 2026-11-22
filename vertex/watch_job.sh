#!/usr/bin/env bash
# Watch a running Vertex AI custom job from your Mac and send Telegram updates.
# Useful when a job was submitted without TELEGRAM_* set (the job itself then sends nothing).
#
#   ./vertex/watch_job.sh <JOB_ID> [REGION]          # foreground
#   nohup ./vertex/watch_job.sh <JOB_ID> asia-southeast1 > /tmp/watch.log 2>&1 &
#
# Sends: state changes, first training loss, eval_loss (at most every 30 min),
# the job's own milestone lines, and the final result or the error.
# Telegram credentials: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from the environment or
# ~/.telegram.env (two lines: export TELEGRAM_BOT_TOKEN=... / export TELEGRAM_CHAT_ID=...).
set -uo pipefail

JOB_ID="${1:?usage: watch_job.sh <JOB_ID> [REGION]}"
REGION="${2:-asia-southeast1}"
PROJECT="${PROJECT:-project-26d3cc32-df40-47b2-a4a}"
INTERVAL="${INTERVAL:-120}"
EVAL_EVERY="${EVAL_EVERY:-1800}"   # seconds between eval_loss updates
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HOME/.telegram.env" ] && . "$HOME/.telegram.env"
export TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}" TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}"
if [ -z "$TELEGRAM_BOT_TOKEN" ] || [ -z "$TELEGRAM_CHAT_ID" ]; then
  echo "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (env or ~/.telegram.env)" >&2; exit 1
fi
GCLOUD="$(command -v gcloud || echo "$HOME/google-cloud-sdk/bin/gcloud")"
notify() { echo "[$(date -u +%H:%M)] $1"; python3 "$HERE/notify.py" "$1"; }

name="$("$GCLOUD" ai custom-jobs describe "$JOB_ID" --region="$REGION" --project="$PROJECT" --format="value(displayName)" 2>/dev/null)"
[ -n "$name" ] || { echo "job $JOB_ID not found in $REGION" >&2; exit 1; }
notify "👀 Watching ${name} (${REGION}) from your Mac"

last_state=""; saw_loss=""; last_eval_sent=0; seen_milestones=""
while true; do
  state="$("$GCLOUD" ai custom-jobs describe "$JOB_ID" --region="$REGION" --project="$PROJECT" --format="value(state)" 2>/dev/null)"
  # newest 400 lines, put back in chronological order (tail -r = reverse on macOS)
  logs="$("$GCLOUD" logging read "resource.labels.job_id=\"$JOB_ID\"" --project="$PROJECT" --limit=400 \
          --format="value(textPayload)" 2>/dev/null | tail -r)"

  if [ -n "$state" ] && [ "$state" != "$last_state" ]; then
    case "$state" in
      JOB_STATE_PENDING|JOB_STATE_QUEUED) notify "⏳ ${name}: waiting for a GPU" ;;
      JOB_STATE_RUNNING)                  notify "🚀 ${name}: running (setup, then training)" ;;
      *)                                  [ -n "$last_state" ] && notify "ℹ️ ${name}: ${state#JOB_STATE_}" ;;
    esac
    last_state="$state"
  fi

  if [ -z "$saw_loss" ] && echo "$logs" | grep -q "'loss'"; then
    saw_loss=1
    notify "🏋️ ${name}: training started. $(echo "$logs" | grep -o "Training PEFT model[^.]*" | tail -n 1)"
  fi

  now=$(date +%s)
  if [ $((now - last_eval_sent)) -ge "$EVAL_EVERY" ]; then
    ev="$(echo "$logs" | grep -o "'eval_loss': '[0-9.e-]*'.*'epoch': '[0-9.]*'" | tail -n 1)"
    if [ -n "$ev" ]; then
      loss="$(echo "$ev" | grep -o "eval_loss': '[0-9.e-]*" | grep -o "[0-9.e-]*$")"
      ep="$(echo "$ev" | grep -o "epoch': '[0-9.]*" | grep -o "[0-9.]*$")"
      notify "📉 ${name}: eval_loss ${loss} at epoch ${ep}"
      last_eval_sent=$now
    fi
  fi

  # milestone lines the job prints itself
  for pat in "best eval_loss [^)]*" "Evaluating ./peft-text-summary[^ ]* on [0-9]* examples"; do
    while IFS= read -r m; do
      [ -z "$m" ] && continue
      case "$seen_milestones" in *"|$m|"*) continue ;; esac
      seen_milestones+="|$m|"
      notify "📌 ${name}: $m"
    done < <(echo "$logs" | grep -o "$pat" | sort -u)
  done

  case "$state" in
    JOB_STATE_SUCCEEDED)
      scores="$(echo "$logs" | grep -E "^(rouge1|rouge2|rougeL) |^bertscore f1" | tail -n 8)"
      notify "✅ ${name} finished.
${scores}"
      exit 0 ;;
    JOB_STATE_FAILED|JOB_STATE_CANCELLED)
      err="$("$GCLOUD" ai custom-jobs describe "$JOB_ID" --region="$REGION" --project="$PROJECT" --format="value(error.message)" 2>/dev/null)"
      notify "❌ ${name}: ${state#JOB_STATE_}. ${err}
$(echo "$logs" | grep -E "Error|Traceback" | tail -n 5)"
      exit 1 ;;
  esac
  sleep "$INTERVAL"
done
