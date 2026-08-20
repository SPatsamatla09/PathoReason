"""Run the PCam PathoReason tiles through Track B (Sid's GPT4Predictor interface).

Targets the Week 2 predict() interface (commit 1a8d657): free-form generation
with extract_json + a frozen-protocol validator, using the pathoreason schema
{label, confidence, evidence: [{feature, grid_cells, description}]} with the
same eight-feature vocabulary and sixteen grid cells. Raw response text and
top-level key order come back in metadata, so key order IS auditable here,
and temperature is pinned to 1.0 inside predict() -- matching the gemma runs.

Sid's validator RAISES on any frozen-protocol violation rather than recording
it; this runner catches those errors and persists err.raw_response /
err.protocol_violations so a violating response is still auditable instead of
vanishing into a retry loop.

Output goes to pcam/runs/<prompt>_trackB.jsonl; MHIST runs are never touched.

    python3 run_trackB.py --prompt cte_p1 --repo /path/to/PathoReason
    python3 run_trackB.py --check   # validate plumbing without any API call
"""

import argparse
import dataclasses
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
RENDERED = os.path.join(ROOT, "prompts", "rendered_pcam")
MANIFEST = os.path.join(ROOT, "pcam", "selection_manifest.json")
RUNS = os.path.join(ROOT, "pcam", "runs")

TRACK_B_MODEL = "gpt-4o"
IMAGE_DETAIL = "high"
MIN_INTERVAL_S = 2.0
MAX_ATTEMPTS = 4
BACKOFF_BASE_S = 10.0

# PCam uses one call per selected tile for each prompt/seed run.
def jobs_for(tiles):
    return [(t, 1) for t in tiles]


def import_track_b(repo):
    """Load Sid's models.base and models.gpt4_predictor without the package init.

    models/__init__.py pulls in the factory, which eagerly imports the PLIP
    track and with it torch -- a heavy dependency Track B never touches. Loading
    the two needed files by path under a synthetic `models` package runs Sid's
    code byte-for-byte while leaving the torch tracks out of the process.
    """
    import importlib.util
    import types

    pkg = types.ModuleType("models")
    pkg.__path__ = [os.path.join(repo, "models")]
    sys.modules["models"] = pkg

    for name in ("models.base", "models.gpt4_predictor"):
        path = os.path.join(repo, name.replace(".", os.sep) + ".py")
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules["models.gpt4_predictor"].GPT4Predictor


def build_predictor(repo):
    from dotenv import load_dotenv

    # let a key in grid_experiment/.env or <repo>/.env work regardless of CWD
    load_dotenv(os.path.join(ROOT, ".env"))
    load_dotenv(os.path.join(repo, ".env"))

    cls = import_track_b(repo)
    return cls(model=TRACK_B_MODEL, image_detail=IMAGE_DETAIL, protocol="pcam")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="cte_p1")
    ap.add_argument("--repo", default=os.environ.get("PATHOREASON_REPO", ""))
    ap.add_argument("--check", action="store_true", help="validate setup, no API calls")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not args.repo or not os.path.isdir(os.path.join(args.repo, "models")):
        sys.exit("--repo must point at a PathoReason checkout (models/ not found)")

    prompt_text = open(os.path.join(RENDERED, f"{args.prompt}.txt")).read()
    tiles = json.load(open(MANIFEST))["tiles"]
    jobs = jobs_for(tiles)
    if args.limit:
        jobs = jobs[: args.limit]

    from PIL import Image  # after arg parsing so --help stays fast

    if args.check:
        for t in tiles:
            p = os.path.join(ROOT, t["gridded"])
            Image.open(p).convert("RGB")
        cls = import_track_b(args.repo)
        print(f"OK  prompt {args.prompt} loaded ({len(prompt_text)} chars)")
        print(f"OK  {len(tiles)} gridded tiles open cleanly, {len(jobs)} calls planned")
        print(f"OK  GPT4Predictor imported: {cls.__module__}.{cls.__name__}")
        from dotenv import load_dotenv

        load_dotenv(os.path.join(ROOT, ".env"))
        load_dotenv(os.path.join(args.repo, ".env"))
        key = os.environ.get("OPENAI_API_KEY") or ""
        print(f"{'OK ' if key else 'BLOCKED'} OPENAI_API_KEY {'found' if key else 'missing'}")
        return

    predictor = build_predictor(args.repo)

    os.makedirs(RUNS, exist_ok=True)
    out_path = os.path.join(RUNS, f"{args.prompt}_trackB.jsonl")
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                r = json.loads(line)
                # only successful calls count as done -- an errored record
                # (rate limit, quota, protocol violation) must retry on resume
                if not r.get("error"):
                    done.add((r["image"], r["replicate"]))
            except (json.JSONDecodeError, KeyError):
                pass
        print(f"resuming: {len(done)} successful calls already recorded")

    last = 0.0
    n = 0
    for tile, rep in jobs:
        if (tile["image"], rep) in done:
            continue
        n += 1
        gap = MIN_INTERVAL_S - (time.time() - last)
        if gap > 0 and last:
            time.sleep(gap)
        last = time.time()
        print(f"[{n}] {tile['image']} rep{rep}", file=sys.stderr, flush=True)

        img = Image.open(os.path.join(ROOT, tile["gridded"])).convert("RGB")
        result = err = None
        err_detail = {}
        attempts = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            t0 = time.time()
            try:
                result = predictor.predict(img, prompt_text)
                attempts.append({"attempt": attempt, "elapsed_s": round(time.time() - t0, 2)})
                break
            except Exception as e:  # SDK raises many types; record and retry
                attempts.append(
                    {"attempt": attempt, "error": f"{type(e).__name__}: {e}",
                     "elapsed_s": round(time.time() - t0, 2)}
                )
                err = f"{type(e).__name__}: {e}"
                # Sid's validator attaches the offending response; keep it
                # auditable instead of letting the retry loop swallow it.
                err_detail = {
                    "raw_response": getattr(e, "raw_response", None),
                    "response_id": getattr(e, "response_id", None),
                    "protocol_violations": getattr(e, "protocol_violations", None),
                }
                if attempt < MAX_ATTEMPTS:
                    time.sleep(BACKOFF_BASE_S * (2 ** (attempt - 1)))

        rec = {
            "dataset": "pcam",
            "prompt_id": args.prompt,
            "track": "B",
            "interface": "PathoReason models.gpt4_predictor.GPT4Predictor (1a8d657)",
            "model": TRACK_B_MODEL,
            "image_detail": IMAGE_DETAIL,
            "temperature": 1.0,  # pinned inside predict()
            "image": tile["image"],
            "replicate": rep,
            "label_true": tile["label"],
            "error": None if result else err,
            "meta": {"attempts": attempts},
        }
        if result is not None:
            md = result.metadata
            rec["raw_response"] = md.get("raw_response")
            rec["raw_key_order"] = md.get("top_level_key_order")
            rec["expected_key_order"] = md.get("expected_key_order")
            rec["key_order_valid"] = md.get("key_order_valid")
            # mirror the gemma record shape so score.py works unchanged
            rec["parsed"] = {
                "label": result.label,
                "confidence": md.get("verbalized_confidence"),
                "evidence": [
                    {
                        "feature": e["feature"],
                        "grid_cells_valid": list(e["grid_cells"]),
                        "grid_cells_invalid": [],  # validator raises on any invalid cell
                        "description": e["description"],
                    }
                    for e in result.evidence
                ],
            }
            rec["violations"] = []
            rec["predictor_metadata"] = {
                "response_id": md.get("response_id"),
                "confidence_status": md.get("confidence_status"),
                "usage": md.get("usage"),
            }
        else:
            rec.update(err_detail)

        with open(out_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    print(f"done: {n} new calls -> {out_path}")


if __name__ == "__main__":
    main()
