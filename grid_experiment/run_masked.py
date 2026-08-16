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
MIN_INTERVAL_S = 13.0


def baseline_by_image():
    """Original unmasked cte_p1 answer (replicate 1) per tile."""
    out = {}
    for line in open(BASELINE_RUN):
        r = json.loads(line)
        if r["replicate"] == 1 and r.get("parsed"):
            out[r["image"]] = {
                "label": r["parsed"]["label"],
                "confidence": r["parsed"]["confidence"],
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        sys.exit("CEREBRAS_API_KEY is not set")

    spec = yaml.safe_load(open(os.path.join(ROOT, "prompts", "pathoreason.yaml")))
    prompt_text = open(os.path.join(ROOT, "prompts", "rendered", f"{PROMPT_ID}.txt")).read()
    base = baseline_by_image()

    tiles = sorted(os.listdir(MASKED))
    jobs = []
    for t in tiles:
        man = json.load(open(os.path.join(MASKED, t, "manifest.json")))
        for arm in ARMS:
            for occ in OCCLUSIONS:
                jobs.append((t, man, arm, occ))
    if args.limit:
        jobs = jobs[: args.limit]

    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
                if not r.get("error"):
                    done.add((r["tile_dir"], r["arm"], r["occlusion"]))
            except (json.JSONDecodeError, KeyError):
                pass
        print(f"resuming: {len(done)} successful calls already recorded")

    session = requests.Session()
    last = 0.0
    n = 0
    for t, man, arm, occ in jobs:
        if (t, arm, occ) in done:
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

        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    print(f"done: {n} new calls -> {OUT}")


if __name__ == "__main__":
    main()
