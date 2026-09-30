#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
set -a; . ./.env; set +a
export PATHO_BASE_URL=https://openrouter.ai/api/v1 PATHO_MODEL=google/gemma-4-31b-it \
       PATHO_PROVIDER=friendli PATHO_KEY_ENV=OPENROUTER_API_KEY PATHO_MIN_INTERVAL_S=1
H=$(python3 -c "import run_experiment as r; print(r.host_tag())")
ARGS=(--masked-dir "masked/xover_own__${H}_k3" --baseline-run "runs/cte_p1_xover__${H}.jsonl" --out "runs/masking_xover_own__${H}_k3.jsonl")
pids=()
for i in 0 1 2 3 4 5; do python3 run_masked.py "${ARGS[@]}" --shard "$i/6" > "runs/logs/xown_shard_$i.log" 2>&1 & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done
python3 run_masked.py "${ARGS[@]}" > runs/logs/xown_final.log 2>&1
echo "--- xover-own masked exit $? $(date)" >> runs/logs/host_probe.log
