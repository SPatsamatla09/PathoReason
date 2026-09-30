"""Causal masking sweep: does occluding the cited evidence change the answer?

For each tile, sends the masked variants produced by mask.py through the same
model + prompt that produced the citations. Two arms per occlusion type:

  cited           the 3-cell subset of cells the model itself named
  tissue_matched  3 uncited cells chosen to occlude the same amount of tissue

If the explanation is causally faithful, masking cited cells should perturb the
label/confidence more than masking the tissue-matched control. The control arm
is what separates "evidence mattered" from "any occlusion rattles the model".

    python3 run_masked.py                # 20 tiles x 2 arms x 3 occlusions = 120 calls
    python3 run_masked.py --limit 2      # smoke test

Writes runs/masking_cte_p1_k3.jsonl, one record per call, resumable.
"""

import argparse
import json
import os
import sys
import time

from run_experiment import (
    BASE_URL,
    KEY_ENV,
    MAX_TRIES,
    counts_as_try,
    mark_invalid,
    MODEL,
    TEMPERATURE,
    b64_png,
    call,
    extract_json,
    top_level_key_order,
    validate,
)

import requests
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
MASKED = os.path.join(ROOT, "masked", "cte_p1_k3")
RUNS = os.path.join(ROOT, "runs")
PROMPT_ID = "cte_p1"
BASELINE_RUN = os.path.join(RUNS, "cte_p1.jsonl")
OUT = os.path.join(RUNS, "masking_cte_p1_k3.jsonl")

ARMS = ["cited", "tissue_matched"]
OCCLUSIONS = ["mean", "blur", "black"]
MIN_INTERVAL_S = float(os.environ.get("PATHO_MIN_INTERVAL_S", "13.0"))


def baseline_by_image():
    """Original unmasked cte_p1 answer (replicate 1) per tile."""
    out = {}
    for line in open(BASELINE_RUN):
        r = json.loads(line)
        if r["replicate"] == 1 and not r.get("error") and (r.get("parsed") or {}).get("label") in ("HP", "SSA"):
            out[r["image"]] = {
                "label": r["parsed"]["label"],
                "confidence": r["parsed"]["confidence"],
                # which deployment produced the baseline, so the pooled analysis can
                # check that baseline and masked calls came from the same provider
                "base_url": r.get("base_url"), "model": r.get("model"),
                "provider": (r.get("meta") or {}).get("provider"),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    global MASKED, BASELINE_RUN, OUT
    ap.add_argument("--masked-dir", default=None)
    ap.add_argument("--baseline-run", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--shard", default=None,
                    help="i/n: run only tiles whose sorted index %% n == i. Shards are disjoint by tile, "
                         "so n concurrent processes can share one --out file (records are appended "
                         "with a single atomic write each)")
    args = ap.parse_args()
    if args.masked_dir:
        MASKED = os.path.join(ROOT, args.masked_dir)
    if args.baseline_run:
        BASELINE_RUN = os.path.join(ROOT, args.baseline_run)
    if args.out:
        OUT = os.path.join(ROOT, args.out)

    api_key = os.environ.get(KEY_ENV)
    if not api_key:
        sys.exit(f"{KEY_ENV} is not set")

    spec = yaml.safe_load(open(os.path.join(ROOT, "prompts", "pathoreason.yaml")))
    prompt_text = open(os.path.join(ROOT, "prompts", "rendered", f"{PROMPT_ID}.txt")).read()
    base = baseline_by_image()

    entries = sorted(os.listdir(MASKED))
    tiles = [t for t in entries if os.path.isfile(os.path.join(MASKED, t, "manifest.json"))]
    if len(tiles) != len(entries):
        print(f"skipping {len(entries) - len(tiles)} non-tile entries in {MASKED}")
    if args.shard:
        i, k = (int(x) for x in args.shard.split("/"))
        tiles = [t for j, t in enumerate(tiles) if j % k == i]
    jobs = []
    for t in tiles:
        man = json.load(open(os.path.join(MASKED, t, "manifest.json")))
        for arm in ARMS:
            for occ in OCCLUSIONS:
                jobs.append((t, man, arm, occ))
    if args.limit:
        jobs = jobs[: args.limit]

    def scan():
        done, tries = set(), {}
        if os.path.exists(OUT):
            for line in open(OUT):
                try:
                    r = json.loads(line)
                    key = (r["tile_dir"], r["arm"], r["occlusion"])
                except (json.JSONDecodeError, KeyError):
                    continue
                if not r.get("error") and (r.get("parsed") or {}).get("label") in ("HP", "SSA"):
                    done.add(key)
                elif counts_as_try(r.get("error")):
                    tries[key] = tries.get(key, 0) + 1
        return done, tries

    done, tries = scan()
    if done or tries:
        print(f"resuming: {len(done)} valid calls recorded")

    session = requests.Session()
    last = 0.0
    n = 0
    for t, man, arm, occ in jobs:
        if (t, arm, occ) in done or tries.get((t, arm, occ), 0) >= MAX_TRIES:
            continue
        n += 1
        gap = MIN_INTERVAL_S - (time.time() - last)
        if gap > 0 and last:
            time.sleep(gap)
        last = time.time()
        print(f"[{n}] {t} {arm} {occ}", file=sys.stderr, flush=True)

        img_path = os.path.join(MASKED, t, f"{arm}__{occ}.png")
        raw, meta, err = call(session, prompt_text, b64_png(img_path), api_key)

        cs = man["cellsets"]
        rec = {
            "prompt_id": PROMPT_ID,
            "model": MODEL,
            "base_url": BASE_URL,
            "temperature": TEMPERATURE,
            "image": man["image"],
            "tile_dir": t,
            "arm": arm,
            "occlusion": occ,
            "masked_cells": cs[arm]["cells"],
            "subset_n": man["subset_used"],
            "cited_cells_full": man["cited_cells_full"],
            "cited_tissue_pct": cs["cited"]["pct_of_tissue_masked"],
            "control_tissue_pct": cs["tissue_matched"]["pct_of_tissue_masked"],
            "control_overlap_with_cited": cs[arm]["overlap_with_cited"],
            "match_quality": man["match_quality"],
            "baseline": base.get(man["image"]),
            "raw_response": raw,
            "error": err,
            "meta": meta,
        }
        if raw is not None:
            payload, perr = extract_json(raw)
            rec["json_extract_error"] = perr
            rec["raw_key_order"] = top_level_key_order(payload) if payload else None
            if payload:
                try:
                    obj = json.loads(payload)
                    viol, norm = validate(obj, spec)
                    rec["parsed"] = norm
                    rec["violations"] = viol
                except json.JSONDecodeError as e:
                    rec["parsed"] = None
                    rec["violations"] = [f"json decode failed: {e}"]
            mark_invalid(rec)

        # one os.write on an O_APPEND descriptor: concurrent shard processes never interleave lines
        fd = os.open(OUT, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, (json.dumps(rec) + "\n").encode())
        finally:
            os.close(fd)

    print(f"done: {n} new calls -> {OUT}")
    ok, tries = scan()
    missing = [(t, a, o) for t, _, a, o in jobs if (t, a, o) not in ok]
    gave_up = [k for k in missing if tries.get(k, 0) >= MAX_TRIES]
    unfinished = [k for k in missing if k not in set(gave_up)]
    if gave_up:
        print(f"GAVE UP on {len(gave_up)} masked calls after {MAX_TRIES} unusable responses each "
              f"(their tiles are dropped by the pooled analysis as incomplete): {gave_up[:10]}")
    if unfinished:
        print(f"INCOMPLETE: {len(unfinished)} planned masked calls not yet done: {unfinished[:10]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
