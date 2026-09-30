#!/bin/bash
# Masking on the remaining 450 test tiles (runs/masking_ext2/PLAN.md). Resumable end to end.
set -uo pipefail
cd "$(dirname "$0")"
set -a; . ./.env; set +a
export PATHO_BASE_URL=https://openrouter.ai/api/v1 PATHO_MODEL=google/gemma-4-31b-it \
       PATHO_PROVIDER=friendli PATHO_KEY_ENV=OPENROUTER_API_KEY PATHO_MIN_INTERVAL_S=1
H=$(python3 -c "import run_experiment as r; print(r.host_tag())")
L=runs/logs/ext2; mkdir -p "$L"
exec >> >(tee -a "$L/ext2.log") 2>&1
echo "=== $(date) ext2 start on $H"
# 1. baselines: 6 shards on disjoint tile lists, then merge
python3 - <<'PY'
import json
t = json.load(open("runs/.mask_ext2_tiles.json"))["tiles"]
for i in range(6):
    open(f"runs/logs/ext2/tiles_s{i}.txt", "w").write(",".join(t[i::6]))
PY
for pass in 1 2; do
  pids=()
  for i in 0 1 2 3 4 5; do
    python3 run_experiment.py --prompt cte_p1 --full-test --tiles "$(cat $L/tiles_s$i.txt)" --reps 1 \
      --tag "_ext2__${H}__s$i" > "$L/baseline_s$i.log" 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
done
python3 - "$H" <<'PY'
import json, sys, glob
H = sys.argv[1]
tiles = json.load(open("runs/.mask_ext2_tiles.json"))["tiles"]
lines, ok = [], {}
for f in sorted(glob.glob(f"runs/cte_p1_ext2__{H}__s*.jsonl")):
    for l in open(f):
        r = json.loads(l); lines.append(l)
        if r.get("replicate", 1) == 1 and not r.get("error") and (r.get("parsed") or {}).get("label") in ("HP", "SSA"):
            ok.setdefault(r["image"], r)
open(f"runs/cte_p1_ext2__{H}.jsonl", "w").writelines(lines)
missing = [t for t in tiles if t not in ok]
prov = {(r.get("meta") or {}).get("provider") for r in ok.values()}
print(f"baseline merged: {len(ok)}/{len(tiles)} valid, providers {prov}, missing {missing[:10]}")
if missing or prov != {"Friendli"}:
    sys.exit("GATE: baseline incomplete or wrong provider - re-run this script to resume")
PY
[ $? -ne 0 ] && { echo "NOT COMPLETE: baseline gate"; exit 1; }
echo "--- baseline done $(date)"
# 2. masks (area-matched, as the extension) and the type-matched secondary arm
if [ ! -d "masked/ext2__${H}_k3" ]; then
  python3 mask.py --from-run "runs/cte_p1_ext2__${H}.jsonl" --subset 3 --per-tile-seed --out "masked/ext2__${H}_k3" > "$L/masks.log" 2>&1
  python3 finalize_mask_ext.py "masked/ext2__${H}_k3"
  python3 mask_typematched.py --src "masked/ext2__${H}_k3" --out "masked/typematched2__${H}_k3"
fi
echo "--- masks done $(date)"
# 3. masked calls: main arm and type-matched arm concurrently, 6 shards each
A=(--masked-dir "masked/ext2__${H}_k3" --baseline-run "runs/cte_p1_ext2__${H}.jsonl" --out "runs/masking_ext2__${H}_k3.jsonl")
B=(--masked-dir "masked/typematched2__${H}_k3" --baseline-run "runs/cte_p1_ext2__${H}.jsonl" --out "runs/masking_typematched2__${H}_k3.jsonl")
pids=()
for i in 0 1 2 3 4 5; do
  PATHO_MASK_ARMS=cited,tissue_matched python3 run_masked.py "${A[@]}" --shard "$i/6" > "$L/areaA_s$i.log" 2>&1 & pids+=($!)
  PATHO_MASK_ARMS=type_matched python3 run_masked.py "${B[@]}" --shard "$i/6" > "$L/typeB_s$i.log" 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
PATHO_MASK_ARMS=cited,tissue_matched python3 run_masked.py "${A[@]}" > "$L/areaA_final.log" 2>&1; ea=$?
PATHO_MASK_ARMS=type_matched python3 run_masked.py "${B[@]}" > "$L/typeB_final.log" 2>&1; eb=$?
echo "--- masked done $(date) (exit area $ea, type $eb)"
echo "=== $(date) ext2 finished"
