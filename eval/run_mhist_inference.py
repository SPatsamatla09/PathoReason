"""Run PathoReason predictors over the MHIST test set.

Week 2 inference runner.

Features
--------
- Uses the official MHIST test partition.
- Uses Aditya's frozen rendered prompts.
- Draws the required 4x4 grid before inference.
- Uses the shared PathoReason predictor interface.
- Content-addressed caching for safe resume.
- Supports small smoke tests with --limit.
- GPT-4o is blocked unless --allow-api is explicitly supplied.
- Writes raw inference records without modifying Rohan's runs.jsonl.

Example
-------
python3 eval/run_mhist_inference.py \
    --model plip \
    --prompt cte_p1 \
    --seed 42 \
    --limit 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------
# Make repository root importable when this file is executed directly.
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.mhist_data_loader import MHISTDataset
from grid_experiment.grid import draw_grid
from models.factory import build_predictor


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROMPT_DIR = REPO_ROOT / "grid_experiment" / "prompts" / "rendered"
CACHE_ROOT = REPO_ROOT / "results" / "inference_cache"
OUTPUT_ROOT = REPO_ROOT / "results" / "inference"


VALID_PROMPTS = {
    "zeroshot",
    "co_p1",
    "co_p2",
    "co_p3",
    "cte_p1",
    "cte_p2",
    "cte_p3",
    "etc_p1",
    "etc_p2",
    "etc_p3",
}

API_MODELS = {"gpt4", "gpt-4o", "openai"}


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set available random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------

def load_prompt(prompt_id: str) -> str:
    """Load one frozen rendered PathoReason prompt."""

    if prompt_id not in VALID_PROMPTS:
        raise ValueError(
            f"Unknown prompt {prompt_id!r}. "
            f"Expected one of: {sorted(VALID_PROMPTS)}"
        )

    path = PROMPT_DIR / f"{prompt_id}.txt"

    if not path.is_file():
        raise FileNotFoundError(
            f"Rendered prompt not found: {path}"
        )

    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------
# Hashing / caching
# ---------------------------------------------------------------------

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def image_hash(image) -> str:
    """Hash exact RGB pixels used for inference."""

    h = hashlib.sha256()

    h.update(image.mode.encode("utf-8"))
    h.update(str(image.size).encode("utf-8"))
    h.update(image.tobytes())

    return h.hexdigest()


def make_cache_key(
    *,
    image_sha256: str,
    model: str,
    prompt_id: str,
    prompt_sha256: str,
    seed: int,
    protocol_version: int = 1,
) -> str:
    """Create deterministic cache key for one inference request."""

    payload = {
        "schema_version": 1,
        "image_sha256": image_sha256,
        "model": model,
        "prompt_id": prompt_id,
        "prompt_sha256": prompt_sha256,
        "seed": seed,
        "protocol_version": protocol_version,
    }

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def cache_path(cache_key: str) -> Path:
    return CACHE_ROOT / cache_key[:2] / f"{cache_key}.json"


def load_cache(cache_key: str) -> dict[str, Any] | None:
    path = cache_path(cache_key)

    if not path.is_file():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(
    cache_key: str,
    record: dict[str, Any],
) -> None:
    """Atomically save one cached prediction."""

    path = cache_path(cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    temp_path.replace(path)


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------

def json_safe(value: Any) -> Any:
    """Convert common non-JSON objects into serializable values."""

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return str(value)


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def append_jsonl(
    record: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def completed_keys(path: Path) -> set[str]:
    """Read cache keys already written to the output JSONL."""

    done: set[str] = set()

    if not path.is_file():
        return done

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                key = record.get("cache_key")

                if key:
                    done.add(key)

            except json.JSONDecodeError:
                continue

    return done


# ---------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:

    set_seed(args.seed)

    normalized_model = args.model.strip().lower()

    # Version 2 marks the frozen Track-B structured-evidence protocol.
    # Track A remains version 1 so existing PLIP results stay valid.
    protocol_version = 2 if normalized_model in API_MODELS else 1

    # Track A PLIP uses fixed HP/SSA zero-shot class prompts, not the
    # structured VLM prompt conditions used by Track B.
    if normalized_model == "plip":
        args.prompt = "zeroshot"

    # --------------------------------------------------------------
    # Safety: never accidentally spend API money.
    # --------------------------------------------------------------

    if normalized_model in API_MODELS and not args.allow_api:
        raise SystemExit(
            "\nRefusing to call a paid API.\n"
            "Re-run with --allow-api only after the PLIP smoke test succeeds.\n"
        )

    prompt_text = (
        None if normalized_model == "plip"
        else load_prompt(args.prompt)
    )
    prompt_sha = sha256_text(
        prompt_text if prompt_text is not None else "zeroshot"
    )

    print("\nPathoReason Week 2 inference")
    print("---------------------------")
    print(f"model:      {normalized_model}")
    print(f"prompt:     {args.prompt}")
    print(f"seed:       {args.seed}")
    print(f"limit:      {args.limit or 'full test set'}")
    print()

    # --------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------

    dataset = MHISTDataset(partition="test")

    selected_indices = list(range(len(dataset)))

    if args.subset_manifest is not None:
        manifest_path = Path(args.subset_manifest)
        manifest = json.loads(manifest_path.read_text())
        requested_images = list(manifest["images"])

        if len(requested_images) != len(set(requested_images)):
            raise ValueError("subset manifest contains duplicate image IDs")

        wanted = set(requested_images)
        index_by_name = {}

        for i in range(len(dataset)):
            sample_i = dataset[i]
            name = sample_i["image_name"]
            if name in wanted:
                index_by_name[name] = i

        missing = [x for x in requested_images if x not in index_by_name]
        if missing:
            raise ValueError(
                f"subset manifest contains {len(missing)} images not found "
                f"in MHIST test partition: {missing[:10]}"
            )

        # Preserve frozen manifest order.
        selected_indices = [index_by_name[x] for x in requested_images]

        print(
            f"Subset manifest: {manifest_path} "
            f"({len(selected_indices)} images)"
        )

    if args.limit is not None:
        selected_indices = selected_indices[:args.limit]

    total = len(selected_indices)

    print(f"MHIST test samples selected: {total}")

    # --------------------------------------------------------------
    # Predictor
    # --------------------------------------------------------------

    print(f"Loading predictor: {normalized_model} ...")

    predictor = build_predictor(normalized_model)

    print(f"Loaded: {predictor.model_name}")
    print()

    # --------------------------------------------------------------
    # Output file
    # --------------------------------------------------------------

    output_suffix = (
        f"{args.prompt}_seed{args.seed}_v{protocol_version}.jsonl"
        if normalized_model in API_MODELS
        else f"{args.prompt}_seed{args.seed}.jsonl"
    )

    output_path = (
        OUTPUT_ROOT
        / normalized_model
        / output_suffix
    )

    done = completed_keys(output_path)

    if done:
        print(f"Resume: {len(done)} records already present.")
        print()

    cache_hits = 0
    new_predictions = 0
    skipped = 0
    failures = 0
    api_calls_made = 0

    start_all = time.perf_counter()

    # --------------------------------------------------------------
    # Inference loop
    # --------------------------------------------------------------

    for index in range(total):

        sample = dataset[selected_indices[index]]

        clean_image = sample["image"]

        # Aditya's frozen prompts explicitly refer to the 4x4 grid.
        inference_image = draw_grid(clean_image.copy())

        img_sha = image_hash(inference_image)

        key = make_cache_key(
            image_sha256=img_sha,
            model=normalized_model,
            prompt_id=args.prompt,
            prompt_sha256=prompt_sha,
            seed=args.seed,
            protocol_version=protocol_version,
        )

        # Already represented in this experiment output.
        if key in done:
            skipped += 1

            print(
                f"[{index + 1:04d}/{total:04d}] "
                f"{sample['image_name']}  SKIP"
            )

            continue

        cached = load_cache(key)

        if cached is not None:
            cache_hits += 1

            record = dict(cached)
            record["cache_hit"] = True

            append_jsonl(record, output_path)

            print(
                f"[{index + 1:04d}/{total:04d}] "
                f"{sample['image_name']}  CACHE"
            )

            continue

        # ----------------------------------------------------------
        # Actual model call
        # ----------------------------------------------------------

        if normalized_model in API_MODELS:
            if (
                args.max_api_calls is not None
                and api_calls_made >= args.max_api_calls
            ):
                print(
                    "\nAPI call limit reached: "
                    f"{api_calls_made}/{args.max_api_calls}. "
                    "Stopping safely."
                )
                break

            api_calls_made += 1

        t0 = time.perf_counter()

        try:
            if normalized_model == "plip":
                model_prompt = [
                    "H&E histopathology image of a hyperplastic polyp",
                    "H&E histopathology image of a sessile serrated adenoma",
                ]
                model_image = clean_image
            else:
                model_prompt = prompt_text
                model_image = inference_image

            prediction = predictor.predict(
                model_image,
                model_prompt,
            )

            latency = time.perf_counter() - t0

            record = {
                "schema_version": 1,

                "cache_key": key,
                "cache_hit": False,

                "image_id": sample["image_name"],
                "partition": sample["partition"],

                "ground_truth": sample["label_name"],
                "ground_truth_index": sample["label"],

                "ssa_votes": sample["ssa_votes"],
                "annotator_agreement": sample[
                    "majority_agreement"
                ],

                "model": prediction.model_name,
                "model_request_name": normalized_model,

                "track": (
                    "B"
                    if normalized_model in API_MODELS
                    else "A"
                ),

                "prompt_id": args.prompt,
                "prompt_sha256": prompt_sha,

                "seed": args.seed,
                "protocol_version": protocol_version,

                "image_sha256": img_sha,
                "grid_applied": normalized_model != "plip",

                "predicted_label": prediction.label,
                "confidence": prediction.confidence,

                "explanation": prediction.explanation,
                "evidence": json_safe(prediction.evidence),

                "metadata": json_safe(prediction.metadata),

                "latency_seconds": latency,
            }

            # Save cache FIRST so a later crash cannot lose this call.
            save_cache(key, record)

            append_jsonl(record, output_path)

            new_predictions += 1

            conf = (
                "None"
                if prediction.confidence is None
                else f"{prediction.confidence:.4f}"
            )

            print(
                f"[{index + 1:04d}/{total:04d}] "
                f"{sample['image_name']}  "
                f"GT={sample['label_name']}  "
                f"PRED={prediction.label}  "
                f"CONF={conf}  "
                f"{latency:.2f}s"
            )

        except Exception as exc:
            failures += 1

            # Persist failures so paid API responses are never lost.
            failure_record = {
                "schema_version": 1,
                "protocol_version": protocol_version,
                "cache_key": key,
                "image_id": sample["image_name"],
                "ground_truth": sample["label_name"],
                "model_request_name": normalized_model,
                "prompt_id": args.prompt,
                "prompt_sha256": prompt_sha,
                "seed": args.seed,
                "image_sha256": img_sha,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "raw_response": getattr(exc, "raw_response", None),
                "response_id": getattr(exc, "response_id", None),
                "protocol_violations": getattr(
                    exc, "protocol_violations", None
                ),
                "latency_seconds": time.perf_counter() - t0,
            }

            failure_path = (
                OUTPUT_ROOT
                / normalized_model
                / (
                    f"{args.prompt}_seed{args.seed}_v"
                    f"{protocol_version}_failures.jsonl"
                )
            )

            append_jsonl(failure_record, failure_path)

            print(
                f"[{index + 1:04d}/{total:04d}] "
                f"{sample['image_name']}  "
                f"ERROR: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    elapsed = time.perf_counter() - start_all

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    print("\nFinished")
    print("--------")
    print(f"new predictions: {new_predictions}")
    print(f"cache hits:      {cache_hits}")
    print(f"already done:    {skipped}")
    print(f"failures:        {failures}")

    if normalized_model in API_MODELS:
        print(f"paid API calls:  {api_calls_made}")

    print(f"runtime:         {elapsed:.2f} s")
    print(f"output:          {output_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Run PathoReason inference on MHIST."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Predictor registered in models/factory.py.",
    )

    parser.add_argument(
        "--prompt",
        required=True,
        choices=sorted(VALID_PROMPTS),
        help="Frozen PathoReason prompt ID.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N test images.",
    )

    parser.add_argument(
        "--subset-manifest",
        type=str,
        default=None,
        help="JSON manifest containing an 'images' list to evaluate.",
    )

    parser.add_argument(
        "--allow-api",
        action="store_true",
        help="Explicitly permit paid API predictors.",
    )

    parser.add_argument(
        "--max-api-calls",
        type=int,
        default=None,
        help=(
            "Maximum number of NEW paid API calls permitted in this run. "
            "Cache hits and completed records do not count."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())