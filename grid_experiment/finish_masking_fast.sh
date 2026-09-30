#!/bin/bash
# Finish the masking extension with N parallel shard workers (disjoint tiles, shared
# output file with atomic appends), then run the per-sweep and pooled analyses and push.
#   screen -dmS kiranfast caffeinate -dimsu bash -c './finish_masking_fast.sh 6 <clone_dir>'
set -uo pipefail
cd "$(dirname "$0")"
N="${1:-6}"
CLONE="$2"
set -a; . ./.env; set +a
export PATHO_BASE_URL=https://openrouter.ai/api/v1 PATHO_MODEL=google/gemma-4-31b-it \
       PATHO_PROVIDER=friendli PATHO_KEY_ENV=OPENROUTER_API_KEY PATHO_MIN_INTERVAL_S=2
HOST=$(python3 -c "import run_experiment as r; print(r.host_tag())")
exec >> >(tee -a "runs/logs/kiran_batch__${HOST}.log") 2>&1
echo "=== $(date) fast finish: $N shard workers on ${HOST}"
ARGS=(--masked-dir "masked/ext__${HOST}_k3" --baseline-run "runs/cte_p1_ext__${HOST}.jsonl"
      --out "runs/masking_ext__${HOST}_k3.jsonl")
pids=()
for i in $(seq 0 $((N - 1))); do
  python3 run_masked.py "${ARGS[@]}" --shard "$i/$N" > "runs/logs/shard_${i}_of_${N}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "--- shards finished $(date)"
# clean-up pass over every tile: resumes, re-queues anything a shard left unfinished
if python3 run_masked.py "${ARGS[@]}" &&
   python3 analyze_masking.py --run "runs/masking_ext__${HOST}_k3.jsonl"; then
  touch "runs/logs/.done__${HOST}__ext_masked"
  echo "--- done ext_masked $(date)"
else
  echo "NOT COMPLETE: ext_masked (re-run this script to resume)"
fi
python3 analyze_masking_pooled.py
./autopush_when_done.sh "$CLONE"
echo "=== $(date) fast finish complete"
