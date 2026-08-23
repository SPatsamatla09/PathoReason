"""Run PLIP on the 120 PCam causal-masking perturbations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.factory import build_predictor

ROOT = Path(__file__).resolve().parent
MASKED_ROOT = ROOT / "pcam" / "masked"
OUT = ROOT / "pcam" / "runs" / "masking_cte_p1_k3_plip.jsonl"

ARMS = ("cited", "tissue_matched")
OCCLUSIONS = ("mean", "blur", "black")

PCAM_PROMPTS = [
    "H&E histopathology image of normal lymph node tissue without metastatic tumor",
    "H&E histopathology image of lymph node tissue containing metastatic tumor",
]


def load_done():
    done = set()

    if not OUT.exists():
        return done

    for line in OUT.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            if not r.get("error"):
                done.add((r["tile_dir"], r["arm"], r["occlusion"]))
        except (json.JSONDecodeError, KeyError):
            pass

    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    jobs = []

    for tile_dir in sorted(p for p in MASKED_ROOT.iterdir() if p.is_dir()):
        manifest = json.loads((tile_dir / "manifest.json").read_text())

        for arm in ARMS:
            for occ in OCCLUSIONS:
                path = tile_dir / f"{arm}__{occ}.png"

                if not path.is_file():
                    raise FileNotFoundError(path)

                jobs.append((tile_dir.name, manifest, arm, occ, path))

    done = load_done()

    pending = [
        j for j in jobs
        if (j[0], j[2], j[3]) not in done
    ]

    if args.limit is not None:
        pending = pending[:args.limit]

    print(f"total jobs: {len(jobs)}")
    print(f"successful jobs already recorded: {len(done)}")
    print(f"pending jobs selected: {len(pending)}")

    if args.check:
        print("CHECK ONLY")
        return

    predictor = build_predictor("plip")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    for i, (tile, manifest, arm, occ, path) in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] {tile} {arm} {occ}", flush=True)

        rec = {
            "dataset": "pcam",
            "model": "plip",
            "image": manifest["image"],
            "tile_dir": tile,
            "arm": arm,
            "occlusion": occ,
            "masked_cells": manifest["cellsets"][arm]["cells"],
            "match_quality": manifest["match_quality"],
        }

        try:
            img = Image.open(path).convert("RGB")
            pred = predictor.predict(img, PCAM_PROMPTS)

            probs = pred.metadata["class_probabilities"]

            # PLIP internally calls its two slots HP/SSA.
            normal_p = float(probs["HP"])
            tumor_p = float(probs["SSA"])

            rec.update({
                "label": "normal" if normal_p >= tumor_p else "tumor",
                "confidence": max(normal_p, tumor_p),
                "class_probabilities": {
                    "normal": normal_p,
                    "tumor": tumor_p,
                },
                "error": None,
            })

        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"

        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    print(f"done: {len(pending)} new predictions -> {OUT}")


if __name__ == "__main__":
    main()
