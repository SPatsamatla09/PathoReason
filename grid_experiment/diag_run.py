"""Prompt diagnostic runner (runs/prompt_diagnostic/PLAN.md). One process per (model, arm).

    python3 diag_run.py --model openai/gpt-4.1 --provider openai --arm clean_min [--logprobs] [--extra '{"reasoning": {"max_tokens": 0}}']

Arms: clean_cte | grid_min | clean_min | clean_alias (grid_cte is the existing pilot run).
Output: runs/prompt_diagnostic/<model>__<provider>__<arm>.jsonl. Resumable: a tile with a
recorded answer is skipped, and an unusable answer is recorded as such, never re-asked.
Transient HTTP errors are retried with backoff; auth/billing/model errors stop the run.
"""

import argparse
import base64
import hashlib
import json
import math
import os
import re
import sys
import time

import requests

import run_experiment as rx

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "runs", "prompt_diagnostic")
MIN = ("This is a 224x224 pixel H&E-stained image of a colorectal polyp. Is it a hyperplastic polyp (HP) "
       "or a sessile serrated adenoma/lesion (SSA)? Answer with exactly one of: HP, SSA.")
ALIAS = ("This is a 224x224 pixel H&E-stained image of a colorectal polyp. It belongs to one of two classes. "
         "Class A: sessile serrated adenoma/lesion. Class B: hyperplastic polyp. Answer with exactly one letter: A or B.")
ARMS = {"clean_cte": ("clean", "cte"), "grid_min": ("grid", "min"), "clean_min": ("clean", "min"),
        "clean_alias": ("clean", "alias")}


def parse(kind, text):
    t = (text or "").strip()
    if kind == "cte":
        payload, _ = rx.extract_json(t)
        try:
            lab = (json.loads(payload) if payload else {}).get("label")
        except (json.JSONDecodeError, AttributeError):
            lab = None
        return lab if lab in ("HP", "SSA") else None
    if kind == "min":
        m = re.search(r"\b(HP|SSA)\b", t.upper())
        return m.group(1) if m else None
    m = re.search(r"\b([AB])\b", t.upper())
    return {"A": "SSA", "B": "HP"}[m.group(1)] if m else None


def p_ssa_from_logprobs(kind, choice):
    content = ((choice.get("logprobs") or {}).get("content") or [])
    if not content:
        return None
    s = h = 0.0
    for alt in content[0].get("top_logprobs") or []:
        tok = alt["token"].strip().upper()
        pr = math.exp(alt["logprob"])
        if kind == "alias":
            s += pr if tok.startswith("A") else 0
            h += pr if tok.startswith("B") else 0
        else:
            s += pr if tok.startswith("S") else 0
            h += pr if tok.startswith("H") else 0
    return s / (s + h) if s + h > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--logprobs", action="store_true")
    ap.add_argument("--extra", default="")
    args = ap.parse_args()
    image_kind, prompt_kind = ARMS[args.arm]
    prompt = {"cte": open(os.path.join(ROOT, "prompts", "rendered", "cte_p1.txt")).read(),
              "min": MIN, "alias": ALIAS}[prompt_kind]
    extra = json.loads(args.extra) if args.extra else {}
    key = os.environ["OPENROUTER_API_KEY"]
    os.makedirs(OUT, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{args.model}__{args.provider}__{args.arm}")
    path = os.path.join(OUT, tag + ".jsonl")
    done = {json.loads(l)["image"] for l in open(path)} if os.path.exists(path) else set()
    meta = {t["image"]: t for t in rx.load_full_test()}
    tiles = json.load(open(os.path.join(ROOT, "runs", ".pilot100_tiles.json")))["tiles"]
    session = requests.Session()
    for n, img in enumerate(tiles, 1):
        if img in done:
            continue
        tile = meta[img]
        b64 = (rx.b64_gridded_tile(tile) if image_kind == "grid"
               else base64.b64encode(open(os.path.join(ROOT, "..", "images", img), "rb").read()).decode())
        body = {"model": args.model, "temperature": 1.0,
                "max_tokens": 1500 if prompt_kind == "cte" else 20,
                "provider": {"order": [args.provider], "allow_fallbacks": False},
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                    {"type": "text", "text": prompt}]}], **extra}
        if args.logprobs:
            body.update({"logprobs": True, "top_logprobs": 10})
        d, attempts = None, []
        for attempt in range(1, 7):
            try:
                r = session.post("https://openrouter.ai/api/v1/chat/completions",
                                 headers={"Authorization": f"Bearer {key}"}, json=body, timeout=180)
            except requests.RequestException as e:
                attempts.append({"attempt": attempt, "error": type(e).__name__})
                time.sleep(10 * attempt)
                continue
            attempts.append({"attempt": attempt, "status": r.status_code})
            if r.status_code in (401, 402, 403, 404):
                sys.exit(f"FATAL {r.status_code}: {r.text[:300]}")
            if r.status_code == 200:
                d = r.json()
                if "choices" in d:
                    break
            time.sleep(min(120, 10 * 2 ** (attempt - 1)))
        choice = (d or {}).get("choices", [{}])[0]
        text = (choice.get("message") or {}).get("content")
        rec = {"arm": args.arm, "image_kind": image_kind, "prompt_kind": prompt_kind,
               "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
               "model": args.model, "provider_pinned": args.provider, "provider_served": (d or {}).get("provider"),
               "extra_body": extra or None, "temperature": 1.0, "image": img,
               "label_true": tile["label"], "agreement_band": tile["agreement_band"],
               "raw_response": text, "label": parse(prompt_kind, text),
               "p_ssa_logprob": p_ssa_from_logprobs(prompt_kind, choice) if args.logprobs else None,
               "usage": (d or {}).get("usage"), "attempts": attempts,
               "error": None if d and "choices" in d else "no_response"}
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[{n}] {tag} {img} -> {rec['label']}", file=sys.stderr, flush=True)
        time.sleep(0.3)


if __name__ == "__main__":
    main()
