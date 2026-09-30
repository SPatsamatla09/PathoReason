#!/bin/bash
# Competence pilot: 3 candidate VLMs x 100 representative test tiles x cte_p1 x 1 rep.
# Each model pinned to one first-party provider (fallbacks off). Resumable.
set -uo pipefail
cd "$(dirname "$0")"
set -a; . ./.env; set +a
TILES=$(python3 -c "import json; print(','.join(json.load(open('runs/.pilot100_tiles.json'))['tiles']))")
export PATHO_BASE_URL=https://openrouter.ai/api/v1 PATHO_KEY_ENV=OPENROUTER_API_KEY PATHO_MIN_INTERVAL_S=1
run() {  # model provider extra_body
  local tag
  tag=$(PATHO_MODEL="$1" PATHO_PROVIDER="$2" PATHO_EXTRA_BODY="$3" python3 -c "import run_experiment as r; print(r.host_tag())")
  PATHO_MODEL="$1" PATHO_PROVIDER="$2" PATHO_EXTRA_BODY="$3" \
    python3 run_experiment.py --prompt cte_p1 --full-test --tiles "$TILES" --reps 1 --tag "_pilot__${tag}" \
    > "runs/logs/pilot_${tag}.log" 2>&1
  echo "--- pilot ${tag} exit $? $(date)" >> runs/logs/pilot.log
}
run google/gemini-2.5-flash google-ai-studio '{"reasoning": {"max_tokens": 0}}' &
run qwen/qwen3-vl-235b-a22b-instruct alibaba '' &
run openai/gpt-4.1 openai '' &
wait
echo "=== pilot done $(date)" >> runs/logs/pilot.log
