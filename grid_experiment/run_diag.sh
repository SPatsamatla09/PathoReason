#!/bin/bash
# Prompt diagnostic (runs/prompt_diagnostic/PLAN.md): 10 (model, arm) processes in parallel.
set -uo pipefail
cd "$(dirname "$0")"
set -a; . ./.env; set +a
L=runs/logs/diag
mkdir -p "$L"
for arm in clean_cte grid_min clean_min clean_alias; do
  lp=""; [[ "$arm" == *min* || "$arm" == *alias* ]] && lp="--logprobs"
  python3 diag_run.py --model openai/gpt-4.1 --provider openai --arm "$arm" $lp > "$L/gpt41_$arm.log" 2>&1 &
  python3 diag_run.py --model qwen/qwen3-vl-235b-a22b-instruct --provider alibaba --arm "$arm" > "$L/qwen_$arm.log" 2>&1 &
done
python3 diag_run.py --model google/gemini-2.5-flash --provider google-ai-studio --arm clean_min --extra '{"reasoning": {"max_tokens": 0}}' > "$L/gemini_clean_min.log" 2>&1 &
python3 diag_run.py --model google/gemma-4-31b-it --provider friendli --arm clean_min > "$L/gemma_clean_min.log" 2>&1 &
wait
echo "=== diag done $(date)" >> "$L/diag.log"
