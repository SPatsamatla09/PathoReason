from __future__ import annotations

import random
from pathlib import Path

from data.mhist_data_loader import MHISTDataset
from models.factory import build_predictor

from runs_schema import RunRecord, Track, append_run, load_runs
from confidence_audit import audit_from_runs, audit_report

PROMPT_DIR = Path(__file__).resolve().parent.parent / "grid_experiment" / "prompts" / "rendered"


def load_prompt_text(prompt_id: str) -> str:
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


def generate_audit_data(n: int = 100, prompt_id: str = "cte_p1", results_path: str = "results/runs.jsonl", seed: int = 0):
    predictor = build_predictor("gpt-4o")
    prompt_text = load_prompt_text(prompt_id)
    samples = load_mhist_audit_sample(n=n, seed=seed)

    ground_truth = {}
    for image_id, image, true_label in samples:
        ground_truth[image_id] = true_label
        result = predictor.predict(image, prompt_text)
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
    return ground_truth


def run_pipeline(n: int = 100, prompt_id: str = "cte_p1", results_path: str = "results/runs.jsonl", seed: int = 0):
    ground_truth = generate_audit_data(n=n, prompt_id=prompt_id, results_path=results_path, seed=seed)
    runs_df = load_runs(results_path)
    result = audit_from_runs(runs_df, ground_truth, track_value="B")
    print(audit_report(result))
    return result


if __name__ == "__main__":
    run_pipeline()
