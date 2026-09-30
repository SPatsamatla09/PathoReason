"""Run the ordering-control conditions on the abl100 tile set.

Conditions (prompt design: render_ordering_controls.py; analysis plan:
runs/ordering_controls/PLAN.md):

  co              classify-only baseline (prompts/rendered/co_p1.txt)
  cte             classify-then-explain (cte_p1.txt): reference for the 41-0 contrast
  etc             explain-then-classify (etc_p1.txt): the effect being explained
  ins_filler      copy a non-diagnostic filler paragraph before the label
  ins_checklist   copy a direction-symmetric polyp checklist (HP-first or SSA-first, 50/50)
  ins_copied      copy a DIFFERENT tile's explanation before the label
  ins_self        copy THIS tile's own explanation before the label
  describe_first  plain-visual description of this tile before the label

Everything runs on ONE host (PATHO_BASE_URL / PATHO_MODEL / PATHO_PROVIDER, see
run_experiment.py); co, cte and etc are re-run there because comparisons must
never cross hosts. Output files never mix hosts:

    runs/ordering_controls/<condition>__<host_tag>.jsonl

Order is tile-major and interleaved: for each replicate, tiles in a seeded
order, and for each tile the conditions in a seeded per-(tile, replicate)
order. Drift in the served model over the run therefore spreads evenly
across conditions instead of aliasing onto whichever condition ran last.

--reps K (default 1) draws K independent samples per tile per condition; the
PLAN pre-registers K=3. Resumable: a (tile, replicate) counts as done only with
a valid HP/SSA label. Unusable responses (unparseable JSON, label not HP/SSA)
are recorded as errors and re-queued within the same invocation, up to
MAX_TRIES per (condition, tile, replicate); transient HTTP failures are retried
without spending that budget. Calls that exhaust it are reported as GIVEN UP
(the analyzer's completeness gate then decides); the runner exits non-zero only
if planned calls remain unattempted or unfinished.

    python3 run_ordering_controls.py --conditions all --reps 3
    python3 run_ordering_controls.py --conditions ins_copied --limit 5     # smoke test
    python3 run_ordering_controls.py --conditions all --dry-run
"""

import argparse
import difflib
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter

import requests
import yaml

import run_experiment as rx
from render_ordering_controls import CRITERION_PARAPHRASES, DIAGNOSTIC_TERMS

ROOT = os.path.dirname(os.path.abspath(__file__))
RENDERED = os.path.join(ROOT, "prompts", "rendered")
CONTROLS = os.path.join(ROOT, "prompts", "rendered_ordering_controls")
OUTDIR = os.path.join(ROOT, "runs", "ordering_controls")
SEED = 20260929
MAX_TRIES = 3

CONDITIONS = ["co", "cte", "etc", "ins_filler", "ins_checklist", "ins_copied", "ins_self", "describe_first"]
COPY_CONDITIONS = {"ins_filler", "ins_checklist", "ins_copied", "ins_self"}


def prompt_for(condition, image, index):
    stem = image.replace(".png", "")
    if condition == "co":
        path = os.path.join(RENDERED, "co_p1.txt")
    elif condition == "cte":
        path = os.path.join(RENDERED, "cte_p1.txt")
    elif condition == "etc":
        path = os.path.join(RENDERED, "etc_p1.txt")
    elif condition == "ins_checklist":
        path = os.path.join(CONTROLS, f"ins_checklist_{index['checklist_variants'][image]}.txt")
    elif condition in ("ins_copied", "ins_self"):
        path = os.path.join(CONTROLS, condition, stem + ".txt")
    else:
        path = os.path.join(CONTROLS, f"{condition}.txt")
    return path, open(path).read()


def inserted_text(prompt):
    m = re.search(r"<<<BEGIN TEXT>>>\n(.*?)\n<<<END TEXT>>>", prompt, re.S)
    return m.group(1) if m else None


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def diagnostic_hits(text, terms=DIAGNOSTIC_TERMS):
    low = (text or "").lower()
    return [t for t in terms if re.search(t, low)]


def valid(r):
    return not r.get("error") and (r.get("parsed") or {}).get("label") in ("HP", "SSA")


def scan(path):
    """(done set of (image, rep) with a valid label, Counter of failed tries)."""
    done, tries = set(), Counter()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r["image"], r.get("replicate", 1))
            if valid(r):
                done.add(key)
            elif rx.counts_as_try(r.get("error")):
                tries[key] += 1
    return done, tries


def build_record(c, tile, rep, ppath, prompt, raw, meta_call, err, spec, index, host):
    rec = {
        "condition": c, "replicate": rep,
        "prompt_file": os.path.relpath(ppath, ROOT),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
        "model": rx.MODEL, "base_url": rx.BASE_URL, "provider_pinned": rx.PROVIDER,
        "host_tag": host, "temperature": rx.TEMPERATURE,
        "image": tile["image"], "label_true": tile["label"],
        "ssa_votes_out_of_7": tile["ssa_votes_out_of_7"], "agreement_band": tile["agreement_band"],
        "n_empty_cells": tile["n_empty_cells"], "empty_cells": tile["empty_cells"],
        "raw_response": raw, "error": err, "meta": meta_call,
    }
    if c == "ins_copied":
        d = index["donors"][tile["image"]]
        rec.update({k: d[k] for k in ("donor_image", "donor_etc_label", "donor_true_label")})
    if c == "ins_self":
        rec["own_etc_label"] = index["ins_self"][tile["image"]]["own_etc_label"]
    if c == "ins_checklist":
        rec["checklist_variant"] = index["checklist_variants"][tile["image"]]
    if raw is None:
        return rec
    payload, perr = rx.extract_json(raw)
    rec["json_extract_error"] = perr
    rec["raw_key_order"] = rx.top_level_key_order(payload) if payload else None
    obj = None
    if payload:
        for path, strict in (("strict", True), ("lenient", False)):
            try:
                obj = json.loads(payload, strict=strict)
                rec["parse_path"] = path
                break
            except json.JSONDecodeError:
                continue
    if not isinstance(obj, dict):
        rec["parsed"] = None
        rec["error"] = rec["error"] or f"parse_failed ({perr or 'json decode'})"
        return rec
    viol, parsed = rx.validate(obj, spec)
    rec["parsed"], rec["violations"] = parsed, viol
    if parsed.get("label") not in ("HP", "SSA"):
        rec["error"] = rec["error"] or f"invalid_label {parsed.get('label')!r}"
    pre = obj.get("preamble")
    rec["preamble"] = pre
    rec["preamble_words"] = len(str(pre or "").split())
    if c in COPY_CONDITIONS:
        rec["copy_similarity"] = round(
            difflib.SequenceMatcher(None, norm(inserted_text(prompt)), norm(pre)).ratio(), 4)
    if c == "describe_first":
        rec["describe_diagnostic_hits"] = diagnostic_hits(pre)
        rec["describe_paraphrase_hits"] = diagnostic_hits(pre, CRITERION_PARAPHRASES)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="all")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="cap tiles (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="build prompts, no API calls")
    args = ap.parse_args()

    conds = CONDITIONS if args.conditions == "all" else [c.strip() for c in args.conditions.split(",")]
    bad = [c for c in conds if c not in CONDITIONS]
    if bad:
        sys.exit(f"unknown conditions: {bad}")

    tile_list = json.load(open(os.path.join(ROOT, "runs", ".abl100_tiles.json")))
    meta = {t["image"]: t for t in rx.load_full_test()}
    tiles = [meta[t] for t in tile_list]
    if args.limit:
        tiles = tiles[: args.limit]
    index = json.load(open(os.path.join(CONTROLS, "index.json")))
    spec = yaml.safe_load(open(rx.SPEC))
    host = rx.host_tag()
    outs = {c: os.path.join(OUTDIR, f"{c}__{host}.jsonl") for c in conds}

    if args.dry_run:
        for c in conds:
            sizes = [len(prompt_for(c, t["image"], index)[1]) for t in tiles]
            print(f"{c:15s} {len(tiles)} tiles x {args.reps} reps, prompts {min(sizes)}-{max(sizes)} chars "
                  f"-> {os.path.relpath(outs[c], ROOT)}")
        print(f"total planned calls: {len(conds) * len(tiles) * args.reps}")
        return

    api_key = os.environ.get(rx.KEY_ENV)
    if not api_key:
        sys.exit(f"{rx.KEY_ENV} is not set")
    os.makedirs(OUTDIR, exist_ok=True)

    def build_plan():
        state = {c: scan(outs[c]) for c in conds}
        plan = []
        for rep in range(1, args.reps + 1):
            order = tiles[:]
            random.Random(f"{SEED}:tiles:{rep}").shuffle(order)
            for tile in order:
                cs = conds[:]
                random.Random(f"{SEED}:{tile['image']}:{rep}").shuffle(cs)
                for c in cs:
                    done, tries = state[c]
                    key = (tile["image"], rep)
                    if key in done or tries[key] >= MAX_TRIES:
                        continue
                    plan.append((c, tile, rep))
        return plan

    session = requests.Session()
    last = 0.0
    # first pass runs the plan; later passes re-queue anything that came back
    # unusable or failed transiently, so the run finishes unattended
    for pass_no in range(1, MAX_TRIES + 2):
        plan = build_plan()
        if not plan:
            break
        print(f"pass {pass_no}: {len(plan)} calls to run on {host}", file=sys.stderr, flush=True)
        for n, (c, tile, rep) in enumerate(plan, 1):
            gap = rx.MIN_INTERVAL_S - (time.time() - last)
            if gap > 0 and last:
                time.sleep(gap)
            last = time.time()
            ppath, prompt = prompt_for(c, tile["image"], index)
            print(f"[{n}/{len(plan)}] {c} {tile['image']} rep{rep}", file=sys.stderr, flush=True)
            raw, meta_call, err = rx.call(session, prompt, rx.b64_gridded_tile(tile), api_key)
            rec = build_record(c, tile, rep, ppath, prompt, raw, meta_call, err, spec, index, host)
            with open(outs[c], "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    report, unfinished = {}, 0
    for c in conds:
        done, tries = scan(outs[c])
        missing = [(t["image"], rep) for rep in range(1, args.reps + 1) for t in tiles
                   if (t["image"], rep) not in done]
        gave_up = [k for k in missing if tries[k] >= MAX_TRIES]
        if missing:
            report[c] = {"missing": len(missing), "gave_up": len(gave_up), "examples": missing[:5]}
            unfinished += len(missing) - len(gave_up)
    if report:
        print("NOT COMPLETE:", json.dumps(report, indent=1))
    if unfinished:
        sys.exit(1)
    print(f"runner finished: {len(conds)} conditions x {len(tiles)} tiles x {args.reps} reps on {host}"
          + (" (some calls given up; see above)" if report else ""))


if __name__ == "__main__":
    main()
