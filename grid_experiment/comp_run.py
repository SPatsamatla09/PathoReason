"""Competence search runner (runs/competence/PLAN.md). Reuses the pipeline's cte_p1 prompt and grid render;
only the class block, label tokens and optional few-shot examples change.

    python3 comp_run.py --model google/gemma-4-31b-it --provider friendli --config neutral_cte --tiles screen [--limit 5]
    configs:  co_p1 | cte_p1 | neutral_cte | names_crit_fs | neutral_crit_fs     tiles: screen | dev_rest | dev | test
    controls: --control none | noimage | mismatch

Local model (llama-server, OpenAI-compatible): set COMP_BASE_URL=http://127.0.0.1:8089/v1 and pass --provider local.
No provider-routing field is sent, no API key is needed and nothing leaves this machine.

Resumable; one record per tile. Spend is logged per call (OpenRouter usage.cost). Keys come from the environment only.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import random
import re
import sys
import time

import requests

import run_experiment as rx

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "runs", "competence")
SEED = 20261003
BASE_URL = os.environ.get("COMP_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
LOCAL = BASE_URL.startswith(("http://127.0.0.1", "http://localhost"))
PIPELINE_PROMPTS = {c: open(os.path.join(ROOT, "prompts", "rendered", f"{c}.txt")).read() for c in ("co_p1", "cte_p1")}

# Criteria (WHO 2019-based), parallel structure, similar length, no disease names.
CRIT = {
    "HP": ("Serration is limited to the upper two-thirds of the crypts. Crypts are straight and evenly spaced with "
           "narrow bases, run perpendicular to the muscularis mucosae, and show no dilation, branching or horizontal "
           "growth at the base."),
    "SSA": ("Serration extends down into the base of the crypts. Crypts are distorted with dilated bases, some branch "
            "or grow horizontally along the muscularis mucosae, and form L-shaped or inverted-T shapes at the base."),
}
BASE = open(os.path.join(ROOT, "prompts", "rendered", "cte_p1.txt")).read()
OLD_CLASSES = ("tile from a colorectal polyp. Assign it to exactly one of two classes:\n\n"
               "  HP   hyperplastic polyp\n"
               "  SSA  sessile serrated adenoma (also called sessile serrated lesion)\n")
OLD_TASK = "Classify this tile as HP or SSA."
OLD_LABEL = '"label": "HP" or "SSA",'
for s in (OLD_CLASSES, OLD_TASK, OLD_LABEL):
    assert BASE.count(s) == 1, s


def wrap(text, width=78, indent=11):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width - indent:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return ("\n" + " " * indent).join(lines)


def prompt_neutral(a_is):
    """a_is: 'HP' or 'SSA' -- which criteria Class A carries."""
    b_is = "SSA" if a_is == "HP" else "HP"
    block = ("tile from a colorectal polyp. Assign it to exactly one of two classes, defined\nby these criteria:\n\n"
             f"  Class A  {wrap(CRIT[a_is])}\n\n  Class B  {wrap(CRIT[b_is])}\n")
    p = BASE.replace(OLD_CLASSES, block).replace(OLD_TASK, "Classify this tile as Class A or Class B.")
    p = p.replace(OLD_LABEL, '"label": "A" or "B",')
    assert "hyperplastic" not in p.lower() and "sessile serrated" not in p.lower()
    return p


def prompt_names():
    block = ("tile from a colorectal polyp. Assign it to exactly one of two classes, defined\nby these criteria:\n\n"
             f"  HP   hyperplastic polyp. {wrap(CRIT['HP'], indent=7)}\n\n"
             f"  SSA  sessile serrated adenoma (also called sessile serrated lesion). {wrap(CRIT['SSA'], indent=7)}\n")
    return BASE.replace(OLD_CLASSES, block)


def labels():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "..", "annotations.csv"))))
    return ({r["Image Name"]: r["Majority Vote Label"] for r in rows},
            {r["Image Name"]: int(r["Number of Annotators who Selected SSA (Out of 7)"]) for r in rows},
            {r["Image Name"]: r["Partition"] for r in rows})


def ab_assignment():
    """Fixed per-tile Class-A assignment, 50/50 within label x band strata (all tiles, seeded)."""
    path = os.path.join(OUT, "ab_assignment.json")
    if os.path.exists(path):
        return json.load(open(path))
    lab, votes, _ = labels()
    by = {}
    for n in sorted(lab):
        by.setdefault((lab[n], rx.band(votes[n])), []).append(n)
    rng, out = random.Random(SEED + 7), {}
    for k in sorted(by):
        names = by[k][:]
        rng.shuffle(names)
        for i, n in enumerate(names):
            out[n] = "HP" if i % 2 == 0 else "SSA"
    json.dump(out, open(path, "w"))
    return out


def fewshot_examples():
    """6 examples (3 HP, 3 SSA) from unanimous-agreement tiles of fewshot_pool, seeded, fixed for all calls."""
    path = os.path.join(OUT, "fewshot_examples.json")
    if os.path.exists(path):
        return json.load(open(path))
    lab, votes, _ = labels()
    pool = json.load(open(os.path.join(OUT, "splits.json")))["fewshot_pool"]
    rng = random.Random(SEED + 11)
    hp = sorted(n for n in pool if lab[n] == "HP" and votes[n] == 0)
    ssa = sorted(n for n in pool if lab[n] == "SSA" and votes[n] == 7)
    ex = rng.sample(hp, 3) + rng.sample(ssa, 3)
    rng.shuffle(ex)
    out = [{"image": n, "label": lab[n]} for n in ex]
    json.dump(out, open(path, "w"), indent=1)
    return out


def b64_grid(name):
    return rx.b64_gridded_tile({"image": name, "gridded": None})


THINK_END, SPECIALS = "<unused95>", ("<end_of_turn>", "<eos>", "<unused94>", "<unused95>")
# sampling tiers for the LOCAL model, fixed in the PLAN addendum before any dev call
TIERS = {"1": {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "min_p": 0.0, "repeat_penalty": 1.0},
         "2": {"temperature": 0.0, "top_k": 64, "top_p": 0.95, "min_p": 0.0, "repeat_penalty": 1.0}}
CELL = re.compile(r"^[A-D][1-4]$")


def strip_thinking(text):
    """MedGemma 1.5 may emit '<unused94>thought ... <unused95>' before its answer (server runs with --special).
    Returns (answer_text, had_trace). An unclosed trace leaves nothing to parse."""
    t = text or ""
    had = "<unused94>" in t or THINK_END in t
    if THINK_END in t:
        t = t.rsplit(THINK_END, 1)[1]
    elif "<unused94>" in t:
        t = ""
    for sp in SPECIALS:
        t = t.replace(sp, "")
    return t.strip(), had


def parse(text, config, a_is):
    text, _ = strip_thinking(text)
    payload, _ = rx.extract_json((text or "").strip())
    try:
        obj = json.loads(payload) if payload else None
    except json.JSONDecodeError:
        obj = None
    if not isinstance(obj, dict):
        return None, None
    lab = str(obj.get("label", "")).strip().upper().replace("CLASS", "").strip()
    if config.startswith("neutral"):
        if lab not in ("A", "B"):
            return None, obj
        return (a_is if lab == "A" else ("SSA" if a_is == "HP" else "HP")), obj
    return (lab if lab in ("HP", "SSA") else None), obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--config", required=True,
                    choices=["co_p1", "cte_p1", "neutral_cte", "names_crit_fs", "neutral_crit_fs"])
    ap.add_argument("--shard", default=None, help="i/n: only tiles whose index %% n == i (parallel local slots)")
    ap.add_argument("--tag", default="", help="suffix for the output file, e.g. a runtime/sampling tag")
    ap.add_argument("--tiles", required=True, choices=["screen", "dev_rest", "dev", "test", "smoke"])
    ap.add_argument("--tier", default="1", choices=["1", "2"],
                    help="LOCAL only: 1 = temperature 1.0, 2 = temperature 0 (see PLAN addendum)")
    ap.add_argument("--control", default="none", choices=["none", "noimage", "mismatch"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--extra", default="")
    args = ap.parse_args()
    lab, votes, part = labels()
    splits = json.load(open(os.path.join(OUT, "splits.json")))
    if args.tiles == "smoke":
        # label-free runtime checks only: 20 fewshot_pool tiles (never dev/test, never the six examples)
        exs = {e["image"] for e in fewshot_examples()}
        tiles = random.Random(SEED + 31).sample(sorted(set(splits["fewshot_pool"]) - exs), 20)
    else:
        tiles = (sorted(n for n in lab if part[n] == "test") if args.tiles == "test" else splits[args.tiles])
    if args.shard:
        i, k = (int(x) for x in args.shard.split("/"))
        tiles = [t for j, t in enumerate(tiles) if j % k == i]
    if args.limit:
        tiles = tiles[: args.limit]
    ab = ab_assignment()
    ex = fewshot_examples() if args.config.endswith("_fs") else []
    mismatch = {}
    if args.control == "mismatch":
        base_set = splits["dev"] if args.tiles in ("screen", "dev_rest", "dev") else tiles
        perm = base_set[:]
        rng = random.Random(SEED + 23)
        while True:
            rng.shuffle(perm)
            if all(a != b for a, b in zip(base_set, perm)):
                break
        mismatch = dict(zip(base_set, perm))
    key = "local" if LOCAL else os.environ["OPENROUTER_API_KEY"]
    if LOCAL != (args.provider == "local"):
        sys.exit("--provider local must be used with a localhost COMP_BASE_URL, and only with it")
    extra = json.loads(args.extra) if args.extra else {}
    sampling = TIERS[args.tier] if LOCAL else None
    sampling_tag = f"tier{args.tier}-t{TIERS[args.tier]['temperature']:g}" if LOCAL else None
    out_dir = os.path.join(OUT, "smoke") if args.tiles == "smoke" else OUT   # smoke files are never scored
    os.makedirs(out_dir, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-",
                 f"{args.config}__{args.model}__{args.provider}{('-' + sampling_tag) if LOCAL else ''}{args.tag}"
                 f"__{args.tiles}__{args.control}")
    path = os.path.join(out_dir, tag + ".jsonl")
    done = {json.loads(l)["image"] for l in open(path)} if os.path.exists(path) else set()
    session, spent, t0, n_new = requests.Session(), 0.0, time.time(), 0
    think_id, marker, root = None, None, BASE_URL.rsplit("/v1", 1)[0]
    if LOCAL:
        marker = session.get(root + "/props", timeout=60).json()["media_marker"]
        tk = session.post(BASE_URL.rsplit("/v1", 1)[0] + "/tokenize",
                          json={"content": "<unused94>", "parse_special": True, "add_special": False}, timeout=60).json()["tokens"]
        if len(tk) != 1:
            sys.exit(f"STOP: <unused94> did not tokenize to one token: {tk}")
        think_id = tk[0]
    for img in tiles:
        if img in done:
            continue
        a_is = ab[img]
        if args.config in PIPELINE_PROMPTS:
            prompt = PIPELINE_PROMPTS[args.config]
        elif args.config in ("neutral_cte", "neutral_crit_fs"):
            prompt = prompt_neutral(a_is)
        else:
            prompt = prompt_names()
        content = []
        if ex:
            content.append({"type": "text", "text": "Here are labelled example tiles of the two classes, drawn with the same grid."})
            for i, e in enumerate(ex, 1):
                if args.config.startswith("neutral"):
                    name = "Class A" if e["label"] == a_is else "Class B"
                else:
                    name = e["label"]
                content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64_grid(e["image"])}})
                content.append({"type": "text", "text": f"Example {i}: {name}"})
            content.append({"type": "text", "text": "Now the tile to classify:"})
        if args.control != "noimage":
            shown = mismatch.get(img, img)
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64_grid(shown)}})
        content.append({"type": "text", "text": prompt})
        body = {"model": args.model, "temperature": 1.0, "max_tokens": 1500,
                "messages": [{"role": "user", "content": content}], **extra}
        seed = None
        if LOCAL:
            seed = int(hashlib.sha256(f"{SEED}:{args.config}:{args.control}:{args.tier}:{img}".encode()).hexdigest()[:8], 16) % (2 ** 31)
            # thinking off: ban the token that opens a trace (PLAN addendum, amendment 1)
            body.update({**sampling, "seed": seed, "logit_bias": [[think_id, False]]})
        else:
            body["provider"] = {"order": [args.provider], "allow_fallbacks": False}
        n_images = sum(1 for c in content if c["type"] == "image_url")
        if LOCAL:
            # Raw /completion with a prompt that is token-identical to HF's Gemma3Processor (verified):
            # text parts stripped and concatenated, each image wrapped in blank lines, no system prompt.
            ps = "<start_of_turn>user\n" + "".join(
                c["text"].strip() if c["type"] == "text" else "\n\n" + marker + "\n\n" for c in content
            ) + "<end_of_turn>\n<start_of_turn>model\n"
            imgs = [c["image_url"]["url"].split(",", 1)[1] for c in content if c["type"] == "image_url"]
            body = {"prompt": {"prompt_string": ps, "multimodal_data": imgs} if imgs else ps, "n_predict": 1500,
                    **sampling, "seed": seed, "logit_bias": [[think_id, False]], **extra}
        d, attempts = None, []
        for attempt in range(1, 3 if LOCAL else 7):
            try:
                r = session.post(f"{root}/completion" if LOCAL else f"{BASE_URL}/chat/completions",
                                 headers={"Authorization": f"Bearer {key}"}, json=body, timeout=900 if LOCAL else 180)
            except requests.RequestException as e:
                attempts.append({"attempt": attempt, "error": type(e).__name__})
                time.sleep(10 * attempt)
                continue
            attempts.append({"attempt": attempt, "status": r.status_code})
            if r.status_code in (401, 402, 403, 404):
                sys.exit(f"FATAL {r.status_code}: {r.text[:200]}")
            if r.status_code == 200 and LOCAL and "content" in r.json():
                j = r.json()   # normalise the /completion response to the chat-completions shape used below
                d = {"choices": [{"message": {"content": j["content"]},
                                  "finish_reason": {"eos": "stop", "limit": "length"}.get(j.get("stop_type"), j.get("stop_type"))}],
                     "usage": {"prompt_tokens": j.get("tokens_evaluated"), "completion_tokens": j.get("tokens_predicted")},
                     "timings": j.get("timings"), "model": j.get("model"), "truncated": j.get("truncated")}
                break
            if r.status_code == 200 and "choices" in r.json():
                d = r.json()
                break
            if LOCAL:
                sys.exit(f"STOP: local server returned HTTP {r.status_code}: {r.text[:300]}; nothing written")
            time.sleep(min(120, 10 * 2 ** (attempt - 1)))
        if d is None:
            # infrastructure failure: never scored, never recorded as an answer (PLAN addendum)
            sys.exit(f"STOP: no response for {img} after {len(attempts)} attempts ({attempts}); nothing written. Re-run to resume.")
        if LOCAL and ((d.get("usage") or {}).get("prompt_tokens") or 0) < 256 * n_images:
            sys.exit(f"STOP: prompt_tokens {d.get('usage')} < 256 x {n_images} images for {img}: image not ingested")
        text = (d["choices"][0].get("message") or {}).get("content")
        label, obj = parse(text, args.config, a_is)
        answer_text, had_trace = strip_thinking(text)
        cells = sorted({c for e in ((obj or {}).get("evidence") or []) if isinstance(e, dict)
                        for c in (e.get("grid_cells") or []) if isinstance(c, str) and CELL.match(c.strip().upper())}) \
            if isinstance(obj, dict) else []
        cost = ((d or {}).get("usage") or {}).get("cost") or 0
        spent += cost
        n_new += 1
        rec = {"config": args.config, "model": args.model, "provider_pinned": args.provider,
               "provider_served": "local" if LOCAL else (d or {}).get("provider"), "base_url": BASE_URL,
               "served_model": (d or {}).get("model"), "timings": (d or {}).get("timings"),
               "finish_reason": ((d or {}).get("choices") or [{}])[0].get("finish_reason"),
               "tiles": args.tiles, "control": args.control,
               "image": img, "image_shown": None if args.control == "noimage" else mismatch.get(img, img),
               "label_true": lab[img], "ssa_votes": votes[img], "class_a_is": a_is if args.config.startswith("neutral") else None,
               "fewshot": [e["image"] for e in ex] or None,
               "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
               "temperature": sampling["temperature"] if LOCAL else 1.0, "sampling": sampling, "sampling_tag": sampling_tag,
               "seed": seed, "n_images": n_images, "thinking_trace": had_trace, "n_valid_cited_cells": len(cells),
               "thinking_suppressed_token_id": think_id,
               "endpoint": "/completion (HF-identical prompt)" if LOCAL else "/chat/completions",
               "system_fingerprint": (d or {}).get("system_fingerprint"),
               "raw_response": text, "label": label, "parsed_ok": obj is not None,
               "n_evidence": len(obj.get("evidence") or []) if isinstance(obj, dict) else None,
               "usage": (d or {}).get("usage"), "cost_usd": cost, "attempts": attempts,
               "error": None if d else "no_response", "extra_body": extra or None}
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[{n_new}] {tag} {img} -> {label}  (${spent:.4f}, {time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    print(f"done {tag}: {n_new} new calls, ${spent:.4f}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
