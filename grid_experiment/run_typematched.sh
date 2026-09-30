#!/bin/bash
# Tissue-type-matched masking on the Friendli extension tiles: cited arm re-run
# interleaved with the new type_matched arm, same host, same baselines. Waits for the
# host-probe session to finish first. Resumable.
set -uo pipefail
cd "$(dirname "$0")"
while screen -ls 2>/dev/null | grep -qE "\.hostprobe[[:space:]]"; do sleep 20; done
set -a; . ./.env; set +a
export PATHO_BASE_URL=https://openrouter.ai/api/v1 PATHO_MODEL=google/gemma-4-31b-it \
       PATHO_PROVIDER=friendli PATHO_KEY_ENV=OPENROUTER_API_KEY PATHO_MIN_INTERVAL_S=1 \
       PATHO_MASK_ARMS=cited,type_matched
HOST=$(python3 -c "import run_experiment as r; print(r.host_tag())")
exec >> >(tee -a runs/logs/typematched.log) 2>&1
echo "=== $(date) type-matched start on ${HOST}"
ARGS=(--masked-dir "masked/typematched__${HOST}_k3" --baseline-run "runs/cte_p1_ext__${HOST}.jsonl"
      --out "runs/masking_typematched__${HOST}_k3.jsonl")
pids=()
for i in 0 1 2 3 4 5; do python3 run_masked.py "${ARGS[@]}" --shard "$i/6" > "runs/logs/tm_shard_$i.log" 2>&1 & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done
python3 run_masked.py "${ARGS[@]}"; echo "--- type-matched exit $? $(date)"
