#!/bin/bash
# After the protocol training run: README sections 6 and 7, unattended. Pins the final adapter as a private
# dataset and runs the label-free format check (20 training-pool tiles, no label). It never builds or runs a dev
# job: that needs step 3 to be closed and is started by hand.
#   screen -dmS kgafter caffeinate -dimsu bash -c './kaggle_ft/run_after_train.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
export MHIST_DOTENV=
LOGDIR=~/mhist_local/kaggle_stage/logs
LOG=$LOGDIR/after_train.log
mkdir -p "$LOGDIR"
exec >> >(tee -a "$LOG") 2>&1
OUT=~/mhist_local/kaggle_out
PY=~/mhist_local/venv/bin/python
PIP=(--dataset wheels --pip-install 'peft==0.21.2 bitsandbytes==0.50.2 accelerate==1.15.0' --pip-find-links '{wheels}')
SPEED='--attn-implementation sdpa --batch-size 4'      # measured 2026-10-04: same answers as batch 1, 1.43x faster

echo "=== $(date) waiting for the training driver to stop"
until grep -q "driver stops" "$LOGDIR/train_driver.log" 2>/dev/null; do sleep 60; done
run=""
for d in $(ls -d "$OUT"/mhist-priv-train* 2>/dev/null | sort -V); do
  [ -f "$d/step4_lora/train_summary.json" ] && run=$d
done
state=$(python3 -c "import json,sys;s=json.load(open(sys.argv[1]));print(s.get('status'), s.get('protocol_run'), s.get('chosen_epoch'), (s.get('best_adapter') or {}).get('final'))" "$run/step4_lora/train_summary.json" 2>/dev/null || echo none)
echo "--- $(date) newest run: $run -> status, protocol_run, chosen_epoch, final: $state"
case "$state" in
  "complete True "*" True") ;;
  *) echo "=== $(date) STOP: the training run is not a completed protocol run; nothing uploaded"; exit 1 ;;
esac
python3 - "$run/step4_lora/train_summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print("validation per epoch (balanced accuracy, accuracy, SSA call rate):")
for v in [dict(s.get("baseline_untrained_epoch0") or {}, epoch=0)] + list(s.get("validation_per_epoch") or []):
    print(f"  epoch {v.get('epoch')}: bal {v.get('balanced_accuracy')}, acc {v.get('accuracy')}, recall HP {v.get('recall_HP')}, recall SSA {v.get('recall_SSA')}, SSA calls {v.get('ssa_call_rate')}")
print("chosen epoch:", s.get("chosen_epoch"), "| sessions:", s.get("sessions"), "| wall:", s.get("wall_time_seconds"))
PY

echo "--- $(date) pinning the final adapter as a private dataset"
python3 kaggle_ft/kaggle_push.py dataset-create --kind adapter --dir "$run/step4_lora/best_adapter" | grep -v '^+ ' | tail -6
if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "=== $(date) STOP: adapter upload did not finish cleanly"; exit 1; fi

echo "--- $(date) label-free format check (20 pool tiles)"
python3 kaggle_ft/kaggle_push.py kernel-push --script kaggle_ft/infer_jobs.py --slug mhist-priv-format-check \
    --dataset bundle --dataset weights --dataset jobs --dataset adapter "${PIP[@]}" \
    --args="--jobs {jobs}/smoke__cte_p1__none__tier1.jsonl --bundle-dir {bundle} --model-dir {weights} --adapter-dir {adapter} $SPEED" \
    | grep -v '^+ ' | tail -3
if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "=== $(date) STOP: the format-check push failed"; exit 1; fi
python3 kaggle_ft/kaggle_push.py kernel-status --slug mhist-priv-format-check --wait --max-wait-min 120 | grep -v '^+ ' | tail -1
python3 kaggle_ft/kaggle_push.py kernel-output --slug mhist-priv-format-check --force | grep -v '^+ ' | tail -5
$PY kaggle_ft/import_results.py --dry-run --model-tag medgemma-1.5-4b-it-lora-r16 \
    --train-summary "$run/step4_lora/train_summary.json" \
    --results "$OUT"/mhist-priv-format-check/smoke__cte_p1__none__tier1__lora-*.jsonl
rc=$?                                   # read before anything else runs: $(date) would reset $?
echo "=== $(date) format check finished with exit code $rc (0 = ok; dev jobs still need step 3 closed and are not started here)"
python3 kaggle_ft/kaggle_push.py verify-private | grep -v '^+ ' | tail -16
