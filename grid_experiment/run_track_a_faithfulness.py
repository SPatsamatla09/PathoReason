"""Run the feature-specific faithfulness experiment with Track-A PLIP.

Gemma's committed cte_p1 evidence supplies feature names and grid cells only.
PLIP supplies all probabilities used in comprehensiveness and sufficiency.
Images are read from the original Dartmouth MHIST directory; no grid is drawn
and no derived pixels are written to disk. Only a resumable JSONL of numeric
predictions is produced.

Example (from the repository root):

    python3 grid_experiment/run_track_a_faithfulness.py \
      --images-dir /path/to/MHIST/images \
      --annotations /path/to/MHIST/annotations.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grid_experiment.mask import ALL_CELLS, MASKERS, cell_pixel_mask, tissue_per_cell

ROOT = Path(__file__).resolve().parent
DEFAULT_CITATIONS = ROOT / "runs" / "cte_p1.jsonl"
DEFAULT_OUT = ROOT / "runs" / "track_a_faithfulness.jsonl"
SEED = 20260806


def load_annotations(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return {row["Image Name"]: row for row in csv.DictReader(fh)}


def load_features(path: Path) -> list[dict]:
    """Load feature/cell citations from either supported Track-B schema.

    The original Gemma/Cerebras runs store citations under
    ``parsed.evidence`` and use ``image``/``grid_cells_valid``.  Sid's
    full-MHIST GPT-4o inference stores them at top level and uses
    ``image_id``/``grid_cells``.  Normalize both formats here so the masking
    and Track-A probability code below remains identical.
    """
    jobs = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            replicate = record.get("replicate")
            if replicate is not None and replicate != 1:
                continue
            if record.get("error"):
                continue
            image = record.get("image") or record.get("image_id")
            evidence = record.get("evidence")
            if evidence is None:
                evidence = (record.get("parsed") or {}).get("evidence")
            if not image or not evidence:
                continue
            seen = set()
            for item in evidence:
                feature = str(item.get("feature", "")).strip().lower()
                raw_cells = item.get("grid_cells_valid")
                if raw_cells is None:
                    raw_cells = item.get("grid_cells", [])
                cells = sorted({cell for cell in raw_cells if cell in ALL_CELLS})
                key = (feature, tuple(cells))
                if feature and cells and key not in seen:
                    jobs.append({
                        "image": image,
                        "feature": feature,
                        "cells": cells,
                        "citation_model": record.get("model"),
                        "citation_track": record.get("track", "B"),
                        "citation_prompt_id": record.get("prompt_id"),
                        "citation_seed": record.get("seed"),
                    })
                    seen.add(key)
    return jobs


def tissue_matched_random_controls(cited, tissue, n, rng, max_gap_pp=5.0):
    """Sample disjoint same-size controls matched on percent of tile tissue."""
    cited = set(cited)
    k = len(cited)
    pool = [c for c in ALL_CELLS if c not in cited]
    if len(pool) < k:
        return []
    total_tissue = max(sum(tissue.values()), 1)
    target = sum(tissue[c] for c in cited)
    candidates = []
    for combo in itertools.combinations(pool, k):
        gap_pp = 100.0 * abs(sum(tissue[c] for c in combo) - target) / total_tissue
        if gap_pp <= max_gap_pp:
            candidates.append((list(combo), gap_pp))
    rng.shuffle(candidates)
    return sorted(candidates[:n], key=lambda x: tuple(x[0]))


def class_probabilities(predictor, rgb):
    result = predictor.predict(Image.fromarray(rgb.astype(np.uint8), "RGB"), None)
    probs = result.metadata["class_probabilities"]
    return {
        "predicted_label": result.label,
        "predicted_confidence": float(result.confidence),
        "prob_hp": float(probs["HP"]),
        "prob_ssa": float(probs["SSA"]),
    }


def completed_keys(path: Path):
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
                done.add((r["image"], r["feature"], r["variant"], r["occlusion"], r.get("control_index")))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def append(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    # Keep heavyweight model imports out of utility/test imports.
    from models.plip_predictor import PLIPPredictor

    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--random-controls", type=int, default=20)
    ap.add_argument("--max-tissue-gap-pp", type=float, default=5.0)
    ap.add_argument("--max-feature-cells", type=int, default=3, help="use the N highest-tissue cited cells within each feature")
    ap.add_argument("--occlusions", default="mean,blur,black")
    ap.add_argument("--device", default=None, help="cpu, mps, or cuda (automatic by default)")
    ap.add_argument("--limit", type=int, default=None, help="limit image-feature jobs for a smoke test")
    args = ap.parse_args()

    annotations = load_annotations(args.annotations)
    jobs = load_features(args.citations)
    if args.limit:
        jobs = jobs[: args.limit]
    occlusions = [x.strip() for x in args.occlusions.split(",") if x.strip()]
    unknown = set(occlusions) - set(MASKERS)
    if unknown:
        ap.error(f"unknown occlusions: {sorted(unknown)}")

    predictor = PLIPPredictor(device=args.device)
    done = completed_keys(args.out)
    rng = random.Random(SEED)
    baseline_cache = {}

    for number, job in enumerate(jobs, 1):
        image_name, feature, cited_full = job["image"], job["feature"], job["cells"]
        path = args.images_dir / image_name
        if not path.is_file():
            raise FileNotFoundError(f"missing original MHIST image: {path}")
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        if rgb.shape != (224, 224, 3):
            raise ValueError(f"{image_name}: expected 224x224 RGB, got {rgb.shape}")
        tissue = tissue_per_cell(rgb)
        ranked_cited = sorted(cited_full, key=lambda c: (-tissue[c], c))
        k = min(args.max_feature_cells, len(ranked_cited))
        controls = []
        while k >= 1 and not controls:
            cited = sorted(ranked_cited[:k])
            controls = tissue_matched_random_controls(
                cited, tissue, args.random_controls, rng, args.max_tissue_gap_pp
            )
            k -= 1
        if not controls:
            print(f"SKIP {image_name} / {feature}: no disjoint tissue-matched controls")
            continue

        ann = annotations[image_name]
        common = {
            "schema_version": 1, "model": predictor.model_name, "track": "A",
            "image": image_name, "feature": feature, "feature_cells": cited,
            "feature_cells_full": cited_full, "feature_cells_used": len(cited),
            "max_feature_cells": args.max_feature_cells,
            "true_label": ann["Majority Vote Label"],
            "ssa_votes_out_of_7": int(ann["Number of Annotators who Selected SSA (Out of 7)"]),
            "partition": ann["Partition"], "seed": SEED,
            "citation_model": job.get("citation_model"),
            "citation_track": job.get("citation_track"),
            "citation_prompt_id": job.get("citation_prompt_id"),
            "citation_seed": job.get("citation_seed"),
            "n_random_controls_available": len(controls),
            "max_tissue_gap_pp": args.max_tissue_gap_pp,
        }

        if image_name not in baseline_cache:
            baseline_cache[image_name] = class_probabilities(predictor, rgb)
        baseline = baseline_cache[image_name]

        for occ in occlusions:
            masker = MASKERS[occ]
            cited_mask = cell_pixel_mask(cited)
            variants = [("cited_removed", None, cited, masker(rgb, cited_mask))]
            # Sufficiency input: retain cited cells and remove everything else.
            variants.append(("evidence_only", None, cited, masker(rgb, ~cited_mask)))
            for idx, (cells, gap) in enumerate(controls):
                variants.append(("random_control", idx, cells, masker(rgb, cell_pixel_mask(cells))))

            for variant, control_index, cells, variant_rgb in variants:
                key = (image_name, feature, variant, occ, control_index)
                if key in done:
                    continue
                prediction = class_probabilities(predictor, variant_rgb)
                record = {
                    **common, "variant": variant, "occlusion": occ,
                    "control_index": control_index, "masked_cells": cells,
                    "control_tissue_gap_pp": None if control_index is None else controls[control_index][1],
                    "baseline": baseline, **prediction,
                }
                append(args.out, record)
                done.add(key)
        print(f"[{number}/{len(jobs)}] {image_name} / {feature}: {len(controls)} controls")

    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
