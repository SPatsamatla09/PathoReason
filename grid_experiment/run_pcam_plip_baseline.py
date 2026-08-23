"""Run PLIP on the 20 clean PCam images used in the causal masking experiment."""

from __future__ import annotations

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
CLEAN_ROOT = ROOT / "pcam" / "clean"
OUT = ROOT / "pcam" / "runs" / "plip_baseline.jsonl"

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
                done.add(r["image"])
        except (json.JSONDecodeError, KeyError):
            pass

    return done


def main():
    manifests = []

    for tile_dir in sorted(p for p in MASKED_ROOT.iterdir() if p.is_dir()):
        manifest_path = tile_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifests.append(manifest)

    done = load_done()
    pending = [m for m in manifests if m["image"] not in done]

    print(f"total clean images: {len(manifests)}")
    print(f"successful baselines already recorded: {len(done)}")
    print(f"pending: {len(pending)}")

    if not pending:
        return

    predictor = build_predictor("plip")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    for i, manifest in enumerate(pending, 1):
        image_name = manifest["image"]
        path = CLEAN_ROOT / image_name

        print(f"[{i}/{len(pending)}] {image_name}", flush=True)

        rec = {
            "dataset": "pcam",
            "model": "plip",
            "image": image_name,
        }

        try:
            img = Image.open(path).convert("RGB")
            pred = predictor.predict(img, PCAM_PROMPTS)

            probs = pred.metadata["class_probabilities"]

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

    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
