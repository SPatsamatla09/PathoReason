"""Run GPT-4o re-prediction on perturbed PCam images for faithfulness testing.

Uses the same frozen cte_p1 PCam prompt for every perturbation. For each of
20 selected PCam patches, two masking arms (cited and tissue_matched) are
evaluated under three perturbations (mean, blur, black), yielding at most
120 paid calls.

The runner is resumable. Existing successful image/arm/occlusion records are
skipped. Use --check to validate and enumerate jobs without making API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.gpt4_predictor import GPT4Predictor


ROOT = Path(__file__).resolve().parent
PROMPT_ID = "cte_p1"
MASKED_ROOT = ROOT / "pcam" / "masked"
RUNS_ROOT = ROOT / "pcam" / "runs"
BASELINE_RUN = RUNS_ROOT / f"{PROMPT_ID}_trackB.jsonl"
OUT = RUNS_ROOT / f"masking_{PROMPT_ID}_k3.jsonl"
PROMPT_PATH = ROOT / "prompts" / "rendered_pcam" / f"{PROMPT_ID}.txt"

ARMS = ("cited", "tissue_matched")
OCCLUSIONS = ("mean", "blur", "black")


def load_baselines() -> dict[str, dict]:
    """Return the successful unperturbed cte_p1 result for each PCam image."""
    baselines: dict[str, dict] = {}

    if not BASELINE_RUN.exists():
        return baselines

    for line in BASELINE_RUN.read_text().splitlines():
        if not line.strip():
            continue

        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        parsed = row.get("parsed")
        if not parsed or row.get("error"):
            continue

        image = row.get("image")
        if image:
            baselines[image] = {
                "label": parsed.get("label"),
                "confidence": parsed.get("confidence"),
            }

    return baselines


def load_done() -> set[tuple[str, str, str]]:
    """Return successful perturbation jobs already present in the output."""
    done: set[tuple[str, str, str]] = set()

    if not OUT.exists():
        return done

    for line in OUT.read_text().splitlines():
        if not line.strip():
            continue

        try:
            row = json.loads(line)
            if not row.get("error") and row.get("parsed"):
                done.add(
                    (
                        row["tile_dir"],
                        row["arm"],
                        row["occlusion"],
                    )
                )
        except (json.JSONDecodeError, KeyError):
            continue

    return done


def build_jobs() -> list[tuple[str, dict, str, str, Path]]:
    """Enumerate all valid PCam perturbation jobs."""
    if not MASKED_ROOT.is_dir():
        raise FileNotFoundError(
            f"Masked PCam directory not found: {MASKED_ROOT}"
        )

    jobs = []

    for tile_dir in sorted(p for p in MASKED_ROOT.iterdir() if p.is_dir()):
        manifest_path = tile_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing manifest for {tile_dir.name}: {manifest_path}"
            )

        manifest = json.loads(manifest_path.read_text())

        for arm in ARMS:
            for occlusion in OCCLUSIONS:
                image_path = tile_dir / f"{arm}__{occlusion}.png"
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Missing perturbation image: {image_path}"
                    )

                jobs.append(
                    (
                        tile_dir.name,
                        manifest,
                        arm,
                        occlusion,
                        image_path,
                    )
                )

    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate and enumerate jobs without making API calls.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of pending jobs to execute.",
    )
    args = parser.parse_args()

    if not PROMPT_PATH.is_file():
        raise FileNotFoundError(f"Prompt not found: {PROMPT_PATH}")

    prompt_text = PROMPT_PATH.read_text()
    jobs = build_jobs()
    baselines = load_baselines()
    done = load_done()

    pending = [
        job
        for job in jobs
        if (job[0], job[2], job[3]) not in done
    ]

    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative.")
        pending = pending[: args.limit]

    print(f"prompt: {PROMPT_ID}")
    print(f"masked root: {MASKED_ROOT}")
    print(f"total jobs: {len(jobs)}")
    print(f"successful jobs already recorded: {len(done)}")
    print(f"pending jobs selected: {len(pending)}")
    print(f"baseline explanations available: {len(baselines)}")

    if args.check:
        print("CHECK ONLY: no API calls made")
        return

    predictor = GPT4Predictor(
        prompt_text=prompt_text,
        protocol="pcam",
    )

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    completed = 0

    for tile_dir, manifest, arm, occlusion, image_path in pending:
        print(
            f"[{completed + 1}/{len(pending)}] "
            f"{tile_dir} {arm} {occlusion}",
            flush=True,
        )

        record = {
            "dataset": "pcam",
            "prompt_id": PROMPT_ID,
            "image": manifest["image"],
            "tile_dir": tile_dir,
            "arm": arm,
            "occlusion": occlusion,
            "masked_cells": manifest["cellsets"][arm]["cells"],
            "subset_n": manifest["subset_used"],
            "cited_cells_full": manifest["cited_cells_full"],
            "cited_tissue_pct": (
                manifest["cellsets"]["cited"]["pct_of_tissue_masked"]
            ),
            "control_tissue_pct": (
                manifest["cellsets"]["tissue_matched"]["pct_of_tissue_masked"]
            ),
            "control_overlap_with_cited": (
                manifest["cellsets"][arm]["overlap_with_cited"]
            ),
            "match_quality": manifest["match_quality"],
            "baseline": baselines.get(manifest["image"]),
        }

        try:
            prediction = predictor.predict(image_path)
            record.update(prediction)
            record["error"] = None
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["parsed"] = None

        with OUT.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        completed += 1

    print(f"done: {completed} new calls -> {OUT}")


if __name__ == "__main__":
    main()
