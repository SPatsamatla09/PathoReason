#!/bin/bash
# Temperature probe on Friendli (host-default top_p/top_k): does lower effective
# temperature reproduce Cerebras's 99% code-fence rate and lower replicate disagreement?
set -uo pipefail
cd "$(dirname "$0")"
set -a; . ./.env; set +a
export PATHO_BASE_URL=https://openrouter.ai/api/v1 PATHO_MODEL=google/gemma-4-31b-it \
       PATHO_PROVIDER=friendli PATHO_KEY_ENV=OPENROUTER_API_KEY PATHO_MIN_INTERVAL_S=1
ABL=$(python3 -c "import json; print(','.join(json.load(open('runs/.abl100_tiles.json'))))")
python3 run_experiment.py --prompt cte_p1 --full-test --tiles "$ABL" --reps 2 --temperature 0.5 --tag _tempprobe_t05 > runs/logs/tempprobe_t05.log 2>&1 &
python3 run_experiment.py --prompt cte_p1 --full-test --tiles "$ABL" --reps 1 --temperature 0.0 --tag _tempprobe_t0 > runs/logs/tempprobe_t0.log 2>&1 &
wait
echo "--- temp probe done $(date)" >> runs/logs/host_probe.log
