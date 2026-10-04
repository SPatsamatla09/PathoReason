#!/bin/bash
# Waits for the protocol training notebook on Kaggle and, if it paused at its time budget, continues it in a new
# private session (README section 5, "A second training session"). Nothing here looks at dev or test tiles.
# Resumable: start it again at any time; it picks up at the newest mhist-priv-train* notebook in the registry.
#   screen -dmS kgtrain caffeinate -dimsu bash -c './kaggle_ft/run_train_driver.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
export MHIST_DOTENV=
LOG=~/mhist_local/kaggle_stage/logs/train_driver.log
mkdir -p "$(dirname "$LOG")"
exec >> >(tee -a "$LOG") 2>&1
PIP=(--dataset wheels --pip-install 'peft==0.21.2 bitsandbytes==0.50.2 accelerate==1.15.0' --pip-find-links '{wheels}')
PAT='best_adapter|train_summary|train_log|train_steps|split\.json|val_scores|FAILED|model_files'
OUT=~/mhist_local/kaggle_out
MAX_SESSIONS=4

n=1; slug=mhist-priv-train
while [ -d "$OUT/mhist-priv-train$((n + 1))" ] || grep -q "\"[^\"]*/mhist-priv-train$((n + 1))\"" ~/mhist_local/kaggle_stage/registry.json 2>/dev/null; do
  n=$((n + 1)); slug=mhist-priv-train$n
done
echo "=== $(date) training driver start at session $n ($slug)"
while true; do
  python3 kaggle_ft/kaggle_push.py kernel-status --slug "$slug" --wait --max-wait-min 800 | grep -v '^+ ' | tail -2
  python3 kaggle_ft/kaggle_push.py kernel-output --slug "$slug" --force --file-pattern "$PAT" | grep -v '^+ ' | tail -6
  status=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['status'])" "$OUT/$slug/step4_lora/train_summary.json" 2>/dev/null || echo none)
  echo "--- $(date) $slug: train_summary status=$status"
  if [ "$status" != "paused" ]; then
    echo "=== $(date) driver stops: $slug ended with status=$status (complete = ready for the format check; anything else: read the log)"
    break
  fi
  if [ "$n" -ge "$MAX_SESSIONS" ]; then
    echo "=== $(date) driver stops: $MAX_SESSIONS sessions used and still paused"; break
  fi
  prev=$slug; n=$((n + 1)); slug=mhist-priv-train$n
  echo "--- $(date) continuing in a new session: $slug (resumes from $prev)"
  python3 kaggle_ft/kaggle_push.py kernel-push --script kaggle_ft/train_lora.py --slug "$slug" --dataset bundle --dataset weights "${PIP[@]}" \
      --kernel-source "$prev" --args="--model-dir {weights} --data-dir {bundle} --out-dir /kaggle/working/step4_lora --resume-from {input:$prev}" \
      | grep -v '^+ ' | tail -3
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "=== $(date) driver stops: the push of $slug failed"; break
  fi
done
