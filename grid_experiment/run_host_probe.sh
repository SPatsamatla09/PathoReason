#!/bin/bash
# Host-difference investigation, 2026-09-30: (1) sampling probe, (2) crossover of the
# 185 legacy tiles + their original Cerebras masks onto Friendli. Resumable.
set -uo pipefail
cd "$(dirname "$0")"
set -a; . ./.env; set +a
export PATHO_BASE_URL=https://openrouter.ai/api/v1 PATHO_MODEL=google/gemma-4-31b-it \
       PATHO_PROVIDER=friendli PATHO_KEY_ENV=OPENROUTER_API_KEY PATHO_MIN_INTERVAL_S=1
LOG=runs/logs/host_probe.log
exec >> >(tee -a "$LOG") 2>&1
echo "=== $(date) host probe start"
ABL=$(python3 -c "import json; print(','.join(json.load(open('runs/.abl100_tiles.json'))))")
# (1) sampling probe: Gemma's recommended top_p/top_k, K=2, runs alongside the crossover baseline
( PATHO_TOP_P=0.95 PATHO_TOP_K=64 python3 run_experiment.py --prompt cte_p1 --full-test --tiles "$ABL" --reps 2 --tag _decprobe > runs/logs/decprobe.log 2>&1; echo "--- decprobe exit $? $(date)" ) &
P1=$!
# (2a) crossover baselines, host-default sampling
HOST=$(python3 -c "import run_experiment as r; print(r.host_tag())")
python3 run_experiment.py --prompt cte_p1 --full-test --tiles "$(cat runs/.xover_tiles.txt)" --reps 1 --tag "_xover__${HOST}" > runs/logs/xover_baseline.log 2>&1
echo "--- xover baseline exit $? $(date)"
# (2b) crossover masked calls, 6 shards on disjoint tiles
ARGS=(--masked-dir masked/xover_legacy185 --baseline-run "runs/cte_p1_xover__${HOST}.jsonl" --out "runs/masking_xover__${HOST}_k3.jsonl")
pids=()
for i in 0 1 2 3 4 5; do python3 run_masked.py "${ARGS[@]}" --shard "$i/6" > "runs/logs/xover_shard_$i.log" 2>&1 & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done
python3 run_masked.py "${ARGS[@]}" > runs/logs/xover_final.log 2>&1; echo "--- xover masked exit $? $(date)"
wait "$P1"
echo "=== $(date) host probe done"
