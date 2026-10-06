#!/bin/bash
# Step-3 search on the local MedGemma server, exactly as fixed in runs/competence/PLAN.md (addendum + amendments).
# Unattended and resumable: comp_run.py skips tiles already recorded; gates are re-derived from the result files.
#   screen -dmS mgsearch caffeinate -dimsu bash -c './run_medgemma_search.sh'
set -uo pipefail
cd "$(dirname "$0")"
export COMP_BASE_URL=http://127.0.0.1:8089/v1
MODEL=medgemma-1.5-4b-it-Q8_0
LOG=runs/logs/competence/mg_search.log
mkdir -p runs/logs/competence
exec >> >(tee -a "$LOG") 2>&1
echo "=== $(date) MedGemma search start"

run() {  # config tiles control tier
  python3 comp_run.py --model "$MODEL" --provider local --config "$1" --tiles "$2" --control "$3" --tier "$4" \
    >> "runs/logs/competence/mg_$1_$2_$3_t$4.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "STOP: runner exit $rc for $1 $2 $3 tier $4: $(tail -1 "runs/logs/competence/mg_$1_$2_$3_t$4.log" | cut -c1-200)"
    exit 2
  fi
}
gate() {  # config control tier field  -> prints the field (true/false/number/none)
  python3 comp_analyze.py > /dev/null
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import json, sys
cfg, control, tier, field = sys.argv[1:5]
for r in json.load(open("runs/competence/RESULTS.json")):
    if r["config"] == cfg and r["control"] == control and r["provider"].startswith(f"local [tier{tier}-"):
        v = r
        for k in field.split("."):
            v = v.get(k) if isinstance(v, dict) else None
        print(str(v).lower() if isinstance(v, bool) else v)
        break
else:
    print("none")
PY
}

for TIER in 1 2; do
  echo "--- tier $TIER $(date)"
  for CFG in co_p1 cte_p1 neutral_cte neutral_crit_fs; do
    echo "--- [$CFG tier $TIER] screen $(date)"
    run "$CFG" screen none "$TIER"
    echo "    screen: acc $(gate "$CFG" none "$TIER" screen.accuracy), balanced $(gate "$CFG" none "$TIER" screen.balanced_accuracy), parse $(gate "$CFG" none "$TIER" screen.parse_rate)"
    if [ "$(gate "$CFG" none "$TIER" screen_advances)" != "true" ]; then
      echo "    stops at the screen (< 65%)"; continue
    fi
    echo "--- [$CFG tier $TIER] dev_rest $(date)"
    run "$CFG" dev_rest none "$TIER"
    echo "    dev: acc $(gate "$CFG" none "$TIER" dev.accuracy), balanced $(gate "$CFG" none "$TIER" dev.balanced_accuracy)"
    if [ "$(gate "$CFG" none "$TIER" dev_pass)" != "true" ]; then
      echo "    fails dev"; continue
    fi
    if [ "$CFG" = "co_p1" ]; then
      echo "    co_p1 passes dev; it is not a stopping configuration (PLAN addendum); continuing"; continue
    fi
    echo "--- [$CFG tier $TIER] image controls $(date)"
    run "$CFG" dev noimage "$TIER"
    run "$CFG" dev mismatch "$TIER"
    NI=$(gate "$CFG" noimage "$TIER" control_ok_falls_to_chance)
    MM=$(gate "$CFG" mismatch "$TIER" control_ok_falls_to_chance)
    echo "    controls: no-image ok=$NI (bal $(gate "$CFG" noimage "$TIER" dev.balanced_accuracy)), mismatched ok=$MM (bal $(gate "$CFG" mismatch "$TIER" dev.balanced_accuracy))"
    if [ "$NI" = "true" ] && [ "$MM" = "true" ]; then
      echo "{\"config\": \"$CFG\", \"tier\": \"$TIER\", \"model\": \"$MODEL\"}" > runs/competence/STEP3_WINNER.json
      echo "=== $(date) PASS: $CFG tier $TIER passes dev including both controls. Search stops; next is the prereg + one test run."
      exit 0
    fi
    echo "    fails the image controls (holds up without the right image)"
  done
done
echo "=== $(date) no step-3 configuration passed dev (tiers 1 and 2)"
