#!/bin/bash
# One-command chain for Kiran's two experiments on a replacement gemma-4-31b host.
#
#   1. ordering controls: 8 conditions x 100 abl100 tiles x ORDERING_REPS (default 3) = 2,400 calls
#   2. grid-masking extension: 340 baseline calls, a host-agreement gate, then ~6 masked calls per
#      usable tile (~1,900)
#
# Every step is resumable and gated: a step is marked done only when its runner reports a
# complete plan AND its analyzer/gate exits 0. Re-running after a crash, sleep, or quota stall
# skips finished work. Logs, locks and done-markers live in runs/logs/ (inside the repo tree,
# not /tmp, which is wiped on reboot).
#
# Required environment (example for OpenRouter):
#   export PATHO_BASE_URL=https://openrouter.ai/api/v1
#   export PATHO_MODEL=google/gemma-4-31b-it
#   export PATHO_PROVIDER=<upstream slug>          # REQUIRED on OpenRouter: pins one upstream
#   export PATHO_KEY_ENV=OPENROUTER_API_KEY        # and that variable set to the key
#   export PATHO_MIN_INTERVAL_S=2                  # pacing; match the host's rate limit
#
# Launch so it survives the Claude session ending and the Mac idling (nohup does neither here):
#   screen -dmS kiran caffeinate -dimsu bash -c 'cd /Users/adityak/Documents/mhist/grid_experiment && ./run_kiran_batch.sh'
#   screen -r kiran        # attach to watch; Ctrl-A D to detach
set -euo pipefail
cd "$(dirname "$0")"
: "${PATHO_BASE_URL:?set PATHO_BASE_URL}" "${PATHO_MODEL:?set PATHO_MODEL}" "${PATHO_KEY_ENV:?set PATHO_KEY_ENV}"
if [[ "$PATHO_BASE_URL" == *openrouter* && -z "${PATHO_PROVIDER:-}" ]]; then
  echo "PATHO_PROVIDER is required on OpenRouter (one pinned upstream per experiment)"; exit 2
fi
REPS="${ORDERING_REPS:-3}"
HOST=$(python3 -c "import run_experiment as r; print(r.host_tag())")
mkdir -p runs/logs
LOCK="runs/logs/.lock__${HOST}"
if ! mkdir "$LOCK" 2>/dev/null; then echo "another chain for ${HOST} holds ${LOCK}; remove it if stale"; exit 2; fi
# on any exit (including Ctrl-C or the screen session dying) stop child python
# processes before releasing the lock, so a second chain cannot run alongside them
trap 'pkill -P $$ 2>/dev/null || true; rmdir "$LOCK" 2>/dev/null || true' EXIT
trap 'exit 130' INT TERM HUP
LOG="runs/logs/kiran_batch__${HOST}.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date) start, host ${HOST}, ordering reps ${REPS}"

step() { [ -f "runs/logs/.done__${HOST}__$1" ] && { echo "--- skip $1 (done)"; return 1; }; echo "--- run $1 $(date)"; return 0; }
mark() { touch "runs/logs/.done__${HOST}__$1"; echo "--- done $1 $(date)"; }

python3 probe_host.py

# The two experiments are independent: a permanent failure in one must not block the
# other. Each block chains its steps with && (set -e does not apply inside a tested
# function), and a failed block is reported at the end with a non-zero exit.
FAILED=()

# 1. ordering controls: runner exits 1 if planned calls remain unfinished; analyzer exits 1 on a gate
ordering() {
  step ordering || return 0
  python3 run_ordering_controls.py --conditions all --reps "$REPS" &&
  python3 analyze_ordering_controls.py --host "$HOST" &&
  mark ordering
}

# 2. masking extension: fresh classify-then-explain on this host -> gate -> masks -> masked calls
masking() {
  local EXT_TILES
  EXT_TILES=$(python3 -c "import json; print(','.join(json.load(open('runs/.mask_ext_tiles.json'))['tiles']))") || return 1
  if step ext_baseline; then
    python3 run_experiment.py --prompt cte_p1 --full-test --tiles "$EXT_TILES" --reps 1 --tag "_ext__${HOST}" &&
    python3 check_ext_baseline.py "$HOST" &&
    mark ext_baseline || return 1
  fi
  if step ext_masks; then
    rm -rf "masked/ext__${HOST}_k3" "masked/ext__${HOST}_k3_unusable" &&
    python3 mask.py --from-run "runs/cte_p1_ext__${HOST}.jsonl" --subset 3 --per-tile-seed --out "masked/ext__${HOST}_k3" &&
    python3 finalize_mask_ext.py "masked/ext__${HOST}_k3" &&
    mark ext_masks || return 1
  fi
  if step ext_masked; then
    python3 run_masked.py --masked-dir "masked/ext__${HOST}_k3" \
      --baseline-run "runs/cte_p1_ext__${HOST}.jsonl" --out "runs/masking_ext__${HOST}_k3.jsonl" &&
    python3 analyze_masking.py --run "runs/masking_ext__${HOST}_k3.jsonl" &&
    mark ext_masked || return 1
  fi
  python3 analyze_masking_pooled.py
}

ordering || FAILED+=(ordering)
masking || FAILED+=(masking)
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "=== $(date) NOT COMPLETE: ${FAILED[*]} (see messages above; re-run to resume)"
  exit 1
fi
echo "=== $(date) all steps complete"
