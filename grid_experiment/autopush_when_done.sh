#!/bin/bash
# Wait for the "kiran" screen session (run_kiran_batch.sh) to end, then push its
# results to PathoReason through sync_to_pathoreason.py (public-repo-safe checks).
#   screen -dmS kiranpush caffeinate -i bash -c './autopush_when_done.sh <clone_dir>'
cd "$(dirname "$0")"
CLONE="$1"
LOG=runs/logs/autopush.log
exec >>"$LOG" 2>&1
echo "=== $(date) waiting for the kiran session to end"
# match the session name exactly: ".kiran<whitespace>", not ".kiranpush"
while screen -ls 2>/dev/null | grep -qE "\.kiran[[:space:]]"; do sleep 60; done
echo "=== $(date) kiran session ended"
HOST=openrouter__google-gemma-4-31b-it__friendli
if [ -f "runs/logs/.done__${HOST}__ext_masked" ]; then STATUS=complete; else STATUS=INCOMPLETE; fi
BODY=$(mktemp)
python3 - "$STATUS" >"$BODY" <<'PYEOF'
import json, sys
status = sys.argv[1]
print(f"Automated push when the run ended (masking chain {status}). These results have NOT yet")
print("been independently re-verified the way the ordering results were.\n")
try:
    r = json.load(open("runs/masking_pooled_analysis.json"))
    p = r["primary_pooled_unique_tiles"]
    print(f"Pooled tile-level test ({r.get('scope', '')}):")
    print(f"  {p['n_tiles']} unique tiles: {p['tiles_cited_more']} cited-more vs "
          f"{p['tiles_control_more']} control-more, {p['tiles_tied']} tied; sign test p = {p['sign_test_p']}")
    for h, v in (r.get("per_host_unique_tiles") or {}).items():
        if isinstance(v, dict):
            print(f"  host {h[:60]}: {v['n_tiles']} tiles, {v['tiles_cited_more']} vs "
                  f"{v['tiles_control_more']}, p = {v['sign_test_p']}")
    het = r.get("between_host_heterogeneity")
    if isinstance(het, dict):
        print(f"  between-host heterogeneity (Fisher): p = {het['p']} -- {het['reading']}")
    print("\nFiles: runs/masking_pooled_analysis.json, runs/masking_pooled_legacy_analysis.json,")
    print("runs/masking_ext__openrouter__google-gemma-4-31b-it__friendli_k3.jsonl")
except Exception as e:
    print(f"(could not read the pooled analysis: {e})")
PYEOF
cat "$BODY"
python3 sync_to_pathoreason.py "$CLONE" "Add grid-masking extension results (${STATUS}, Friendli host)" "$BODY"
echo "=== $(date) autopush finished (exit $?)"
rm -f "$BODY"
