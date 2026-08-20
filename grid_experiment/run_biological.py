"""Run the biologically-masked variants through gemma with cte_p1.

20 tiles x 3 mask sources (epithelium / stroma / nuclei) x 2 arms
(struct / block-shuffled matched-area control) = 120 calls.

Schema mirrors runs/masking_cte_p1_k3.jsonl plus mask_source,
segmentation_method, named (did gemma cite a feature mapping to this source on
this tile), and the per-mask pixel accounting.

    python3 run_biological.py [--limit N]
"""

import argparse
import json
import os
import sys
import time

import requests
import yaml

from run_experiment import (
    MODEL,
    TEMPERATURE,
    b64_png,
    call,
    extract_json,
    top_level_key_order,
    validate,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
BIO = os.path.join(ROOT, "masked_bio")
OUT = os.path.join(ROOT, "runs", "biological_masking.jsonl")
PROMPT_ID = "cte_p1"
BASELINE_RUN = os.path.join(ROOT, "runs", "cte_p1.jsonl")

SOURCES = ["epithelium", "stroma", "nuclei"]
ARMS = ["struct", "control"]
MIN_INTERVAL_S = 13.0


def baseline_by_image():
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

    tiles = sorted(d for d in os.listdir(BIO) if os.path.isdir(os.path.join(BIO, d)))
    jobs = []
    for t in tiles:
        man = json.load(open(os.path.join(BIO, t, "manifest.json")))
        for src in SOURCES:
            for arm in ARMS:
                jobs.append((t, man, src, arm))
    if args.limit:
        jobs = jobs[: args.limit]

    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
                if not r.get("error"):
                    done.add((r["tile_dir"], r["mask_source"], r["arm"]))
            except (json.JSONDecodeError, KeyError):
                pass
        print(f"resuming: {len(done)} successful calls already recorded")

    session = requests.Session()
    last = 0.0
    n = 0
    for t, man, src, arm in jobs:
        if (t, src, arm) in done:
            continue
        n += 1
        gap = MIN_INTERVAL_S - (time.time() - last)
        if gap > 0 and last:
            time.sleep(gap)
        last = time.time()
        print(f"[{n}] {t} {src} {arm}", file=sys.stderr, flush=True)

        img_path = os.path.join(BIO, t, f"{src}__{arm}.png")
        raw, meta, err = call(session, prompt_text, b64_png(img_path), api_key)

        s = man["sources"][src]
        rec = {
            "prompt_id": PROMPT_ID,
            "model": MODEL,
            "temperature": TEMPERATURE,
            "image": man["image"],
            "tile_dir": t,
            "mask_source": src,
            "arm": arm,
            "named": s["named"],
            "occlusion": man["occlusion"],
            "segmentation_method": man["segmentation_method"] if src == "nuclei" else None,
            "cited_features": man["cited_features"],
            "n_nuclei": man["n_nuclei"] if src == "nuclei" else None,
            "struct_tissue_pct": s["pct_of_tissue"],
            "control_tissue_pct": s["control_pct_of_tissue"],
            "control_overlap_with_struct_pct": s["control_overlap_with_struct_pct"],
            "mask_pct_of_tile": s["pct_of_tile"],
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
