from __future__ import annotations

import random
import time
from pathlib import Path

from data.mhist_data_loader import MHISTDataset
from models.factory import build_predictor

from runs_schema import RunRecord, Track, append_run, load_runs
from confidence_audit import audit_from_runs, audit_report

PROMPT_DIR = Path(__file__).resolve().parent.parent / "grid_experiment" / "prompts" / "rendered"
DEFAULT_PROMPT_ID = "default_gpt4"


def load_prompt_text(prompt_id: str):
    if prompt_id == DEFAULT_PROMPT_ID:
        return None
    return (PROMPT_DIR / f"{prompt_id}.txt").read_text()


def load_mhist_audit_sample(n: int = 100, seed: int = 0):
    dataset = MHISTDataset(partition="test")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    samples = []
    for idx in indices[:n]:
        item = dataset[idx]
        samples.append((item["image_name"], item["image"], item["label_name"]))
    return samples


def predict_with_retry(predictor, image, prompt_text, image_id, max_retries=3, backoff_seconds=2.0):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return predictor.predict(image, prompt_text)
        except Exception as exc:
            last_error = exc
            print(f"  [{image_id}] attempt {attempt}/{max_retries} failed: {exc}")
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)
    print(f"  [{image_id}] giving up after {max_retries} attempts, skipping this image")
    raise last_error


def generate_audit_data(n: int = 100, prompt_id: str = DEFAULT_PROMPT_ID, results_path: str = "results/runs.jsonl", seed: int = 0, max_retries: int = 3):
    predictor = build_predictor("gpt-4o")
    prompt_text = load_prompt_text(prompt_id)
    samples = load_mhist_audit_sample(n=n, seed=seed)

    ground_truth = {}
    failed_images = []
    for i, (image_id, image, true_label) in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] {image_id}")
        try:
            result = predict_with_retry(predictor, image, prompt_text, image_id, max_retries=max_retries)
        except Exception:
            failed_images.append(image_id)
            continue

        ground_truth[image_id] = true_label
        record = RunRecord(
            image_id=image_id,
            model=predictor.model_name,
            track=Track.B,
            prompt_id=prompt_id,
            seed=seed,
            orig_label=result.label,
            orig_conf=result.metadata["verbalized_confidence"],
        )
        append_run(record, path=results_path)

    if failed_images:
        print(f"\n{len(failed_images)} image(s) failed after retries and were skipped: {failed_images}")
    print(f"Succeeded on {len(ground_truth)}/{len(samples)} images.\n")
    return ground_truth


def run_pipeline(n: int = 100, prompt_id: str = DEFAULT_PROMPT_ID, results_path: str = "results/runs.jsonl", seed: int = 0, max_retries: int = 3):
    ground_truth = generate_audit_data(n=n, prompt_id=prompt_id, results_path=results_path, seed=seed, max_retries=max_retries)
    if len(ground_truth) < 30:
        raise RuntimeError(
            f"Only {len(ground_truth)} images succeeded -- too few for a meaningful audit "
            f"(run_confidence_audit expects at least 30). Check the failure messages above."
        )
    runs_df = load_runs(results_path)
    result = audit_from_runs(runs_df, ground_truth, track_value="B")
    print(audit_report(result))
    return result


if __name__ == "__main__":
    run_pipeline()
