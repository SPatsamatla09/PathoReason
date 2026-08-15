"""PathoReason Week 2 cost and runtime dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gpt_experiment(
    path: Path,
    test_size: int,
    input_rate: float,
    output_rate: float,
) -> dict[str, Any]:
    rows = read_jsonl(path)

    input_tokens = 0
    output_tokens = 0
    runtime = 0.0

    for row in rows:
        usage = row.get("metadata", {}).get("usage", {})
        input_tokens += usage.get("input_tokens", 0) or 0
        output_tokens += usage.get("output_tokens", 0) or 0
        runtime += row.get("latency_seconds", 0) or 0

    cost = (
        input_tokens / 1_000_000 * input_rate
        + output_tokens / 1_000_000 * output_rate
    )

    return {
        "experiment": path.stem,
        "type": "API",
        "completed": len(rows),
        "coverage": len(rows) / test_size if test_size else 0,
        "calls": len(rows),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
        "runtime_seconds": runtime,
        "gpu_hours": 0.0,
    }


def baseline_experiment(path: Path) -> dict[str, Any]:
    data = read_json(path)

    runtime = (
        data.get("runtime_seconds")
        or data.get("runtime_sec")
        or data.get("runtime")
        or 0
    )

    gpu_hours = (
        data.get("gpu_hours")
        or data.get("accelerator_hours")
        or data.get("accel_hours")
        or 0
    )

    return {
        "experiment": path.stem,
        "type": "LOCAL",
        "completed": 1,
        "coverage": 1.0,
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "runtime_seconds": float(runtime),
        "gpu_hours": float(gpu_hours),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--test-size", type=int, default=977)

    parser.add_argument(
        "--gpt-input-rate",
        type=float,
        default=2.50,
        help="USD per 1M GPT-4o input tokens.",
    )
    parser.add_argument(
        "--gpt-output-rate",
        type=float,
        default=10.00,
        help="USD per 1M GPT-4o output tokens.",
    )

    args = parser.parse_args()

    experiments: list[dict[str, Any]] = []

    gpt_dir = RESULTS / "inference" / "gpt-4o"

    if gpt_dir.exists():
        for path in sorted(gpt_dir.glob("*_v2.jsonl")):
            experiments.append(
                gpt_experiment(
                    path,
                    args.test_size,
                    args.gpt_input_rate,
                    args.gpt_output_rate,
                )
            )

    baseline_dir = RESULTS / "baselines"

    if baseline_dir.exists():
        for name in [
            "resnet18.json",
            "vit_b16.json",
            "uni.json",
            "virchow.json",
        ]:
            path = baseline_dir / name
            if path.exists():
                experiments.append(baseline_experiment(path))

    total_cost = sum(x["cost_usd"] for x in experiments)
    total_calls = sum(x["calls"] for x in experiments)
    total_gpu = sum(x["gpu_hours"] for x in experiments)

    print("\nPathoReason Cost & Runtime Dashboard")
    print("=" * 94)

    print(
        f"{'Experiment':<30}"
        f"{'Type':<8}"
        f"{'N':>7}"
        f"{'Calls':>8}"
        f"{'Cost':>11}"
        f"{'Runtime':>12}"
        f"{'GPU-h':>10}"
    )

    print("-" * 94)

    for x in experiments:
        runtime_min = x["runtime_seconds"] / 60

        print(
            f"{x['experiment']:<30}"
            f"{x['type']:<8}"
            f"{x['completed']:>7}"
            f"{x['calls']:>8}"
            f"${x['cost_usd']:>9.4f}"
            f"{runtime_min:>10.1f}m"
            f"{x['gpu_hours']:>10.3f}"
        )

    print("-" * 94)

    print(f"Total API calls:       {total_calls}")
    print(f"Total API cost:        ${total_cost:.4f}")
    print(f"Total GPU/accel hours: {total_gpu:.3f}")
    print(f"Configured budget:     ${args.budget:.2f}")
    print(f"Budget remaining:      ${args.budget - total_cost:.4f}")

    # Estimate one full GPT configuration from completed full runs.
    full_api = [
        x for x in experiments
        if x["type"] == "API"
        and x["coverage"] >= 0.95
        and x["calls"] > 0
    ]

    if full_api:
        avg_full_cost = sum(
            x["cost_usd"] / x["coverage"]
            for x in full_api
        ) / len(full_api)

        print(
            f"\nEstimated cost / full GPT configuration: "
            f"${avg_full_cost:.4f}"
        )

        affordable = max(
            0,
            int((args.budget - total_cost) // avg_full_cost),
        )

        print(
            "Additional full GPT configurations affordable: "
            f"{affordable}"
        )

        if total_cost + avg_full_cost > args.budget:
            print(
                "\n⚠ BUDGET FLAG: another full GPT configuration "
                "is projected to exceed the configured budget."
            )
        elif total_cost >= 0.8 * args.budget:
            print(
                "\n⚠ BUDGET FLAG: at least 80% of the configured "
                "budget has been consumed."
            )
        else:
            print("\n✓ Current trajectory is within configured budget.")

    out = RESULTS / "cost_runtime_dashboard.json"

    payload = {
        "budget_usd": args.budget,
        "pricing_usd_per_1m_tokens": {
            "gpt4o_input": args.gpt_input_rate,
            "gpt4o_output": args.gpt_output_rate,
        },
        "total_api_calls": total_calls,
        "total_api_cost_usd": total_cost,
        "total_gpu_hours": total_gpu,
        "experiments": experiments,
    }

    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
