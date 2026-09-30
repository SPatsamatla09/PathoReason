#!/bin/bash
# Prompt-diagnostic addendum controls (PLAN.md addendum): reversed alias and no-image prior.
set -uo pipefail
cd "$(dirname "$0")"
set -a; . ./.env; set +a
L=runs/logs/diag
for arm in clean_alias_rev noimage_min; do
  python3 diag_run.py --model openai/gpt-4.1 --provider openai --arm "$arm" --logprobs > "$L/gpt41_$arm.log" 2>&1 &
  python3 diag_run.py --model qwen/qwen3-vl-235b-a22b-instruct --provider alibaba --arm "$arm" > "$L/qwen_$arm.log" 2>&1 &
done
wait
echo "=== diag2 done $(date)" >> "$L/diag.log"
