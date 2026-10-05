#!/usr/bin/env python3
r"""Merge downloaded Kaggle output files with the local sidecars into comp_run-style records. LOCAL side.

    python3 kaggle_ft/import_results.py --model-tag medgemma-1.5-4b-it-lora-r16 \
        --train-summary ~/mhist_local/kaggle_out/mhist-priv-train/step4_lora/train_summary.json \
        --results ~/mhist_local/kaggle_out/mhist-priv-infer-none-t1/dev__cte_p1__none__tier1__lora-1a2b3c4d.jsonl
    python3 comp_analyze.py                      # scores them with the same parser and the same bars

Each output file comes from kaggle_ft/infer_jobs.py and holds answers only. The labels come from the sidecar
that kaggle_ft/build_dev_jobs.py wrote on this machine (default ~/mhist_local/kaggle_local_only/) and that
never left it.

WHICH MODEL ANSWERED must be stated, one of:
    --train-summary PATH   the answers come from the LoRA adapter of this finished train_lora.py run. Dev answers
                           are accepted only from its FINAL, validation-selected adapter: the summary must say
                           status "complete" and protocol_run true, every answer's adapter hash must equal the
                           summary's best_adapter.adapter_model_sha256, and the adapter's own step4_meta.json
                           (carried in every output line) must say protocol_run true, final true and the chosen
                           epoch. So an epoch checkpoint, the best-so-far adapter of a paused run or a debug run
                           cannot be scored on dev, whatever --model-tag says.
    --base-model           the answers come from the base model with no adapter; any adapter hash is refused.

Records are written under runs/competence/ with comp_run.py's naming scheme, provider `kaggle`:

    <config>__<model-tag>__kaggle-<tier1-t1|tier2-t0>__<screen|dev_rest>__<control>.jsonl

One file for the 100 screen tiles and one for the 200 dev_rest tiles (by runs/competence/splits.json), so that
comp_analyze.py reports the screen and the 300-tile dev result exactly as it does for comp_run.py's files.
--control-tiles dev writes the image controls (noimage, mismatch) the way comp_run.py's search script runs them
instead: one <...>__dev__<control>.jsonl with all 300 tiles in comp_run's order and no separate screen row.
Smoke jobs go to runs/competence/smoke/, which is never scored.

STEP-4 FORMAT GATE. The addendum's pass rule has a part comp_analyze.py does not apply: at least 90% of the 300
dev answers must parse with a valid label AND cite at least one valid grid cell. For every dev run without an
image control this script computes that share over all 300 tiles, prints `STEP-4 FORMAT GATE ... ok / FAILS`
and writes runs/competence/step4_gate__<config>__<model-tag>__<sampling-tag>.json. A step-4 configuration
passes dev only if comp_analyze.py says PASS and this file says "format_ok": true and both controls are OK.
For smoke files (pool tiles, no labels used) the label-free check must pass BEFORE any dev job is run: at
least 90% of the answers (18 of 20) must BOTH end with a normal `stop` AND have a valid label and a valid grid
cell; otherwise the exit code is 1. (PLAN, step-4 Amendment 2, 2026-10-04: the earlier wording required every
answer to end with `stop`; the user set the check to the brief's 90% bar before the final adapter existed.)

LEDGER. Every dev import is appended to runs/competence/step4_imports.ndjson (adapter hash, run signature,
tier, control, time). A second, different adapter for the same configuration, tier and control is refused
(--allow-another-adapter overrides and is recorded): dev is evaluated once per decoding tier.

Each record carries comp_run.py's fields. label, parsed_ok, n_evidence, n_valid_cited_cells and thinking_trace
are computed with comp_run.parse / comp_run.strip_thinking from the raw answer; provider_pinned and
provider_served are `kaggle`; cost_usd is 0. The Kaggle provenance (revision, adapter hash, dtype, batch size,
sampler, library versions, GPU) is kept under `kaggle`.

IT REFUSES TO WRITE ANYTHING if, for any job file involved:
    * a job of the sidecar has no answer, an answer has no job in the sidecar, or a job_id appears twice;
    * the sidecar is not the file build_dev_jobs.py wrote (its sha256 differs from the one in the meta file), or
      a label / vote count in it differs from annotations.csv;
    * tier, seed, token limit or image count of an answer differ from the sidecar;
    * the answers were generated from another version of the job file (jobs_sha256 differs);
    * a prompt shows fewer than 256 tokens per image, or its token count / token-id hash differs from the one
      the official processor gives locally (when the sidecar holds them);
    * the image files were not checked against the job file's sha256 on Kaggle (images_verified is not true),
      unless --allow-unverified-images;
    * a finish_reason is neither `stop` nor `length`, or the banned thinking opener appears in an answer;
    * the answers come from more than one run setting (model, adapter, dtype, batch size, sampler, versions),
      unless --allow-mixed-runs;
    * the weights were not verified against the pinned revision, unless --allow-unverified-weights;
    * the adapter is not the one --train-summary describes, or (dev) not a final protocol-run adapter; a training
      run marked protocol_run=false can be let through with --allow-non-protocol-adapter (exploratory only),
      a non-final adapter cannot;
    * another adapter was already imported for the same configuration, tier and control, unless
      --allow-another-adapter;
    * a record file for the same configuration, model tag, tier and control already exists, unless --force.
"""

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import comp_run as cr          # noqa: E402  parse, strip_thinking, CELL, TIERS, OUT: the scorer's own parser
import infer_jobs as ij        # noqa: E402  tolerant reader, constants (stdlib-only at import)

PROVIDER = "kaggle"
FORMAT_BAR = 0.90              # PLAN, step-4 addendum: valid label and >= 1 valid cited grid cell
LEDGER = "step4_imports.ndjson"   # not *.jsonl: comp_analyze.py reads every *.jsonl in runs/competence/ as records
DEFAULT_LOCAL_DIR = os.path.join(os.environ.get("MHIST_LOCAL") or os.path.expanduser("~/mhist_local"), "kaggle_local_only")


def refuse(problems):
    print("REFUSED, nothing written:", file=sys.stderr)
    for p in problems[:40]:
        print("  - " + p, file=sys.stderr)
    if len(problems) > 40:
        print(f"  ... and {len(problems) - 40} more", file=sys.stderr)
    sys.exit(2)


def sampling_tag(tier):
    return f"tier{tier}-t{cr.TIERS[tier]['temperature']:g}"        # comp_run.py's tag: tier1-t1, tier2-t0


def record_path(out_dir, config, model_tag, tier, tiles, control):
    """comp_run.py's file name, with provider `kaggle`."""
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{config}__{model_tag}__{PROVIDER}-{sampling_tag(tier)}__{tiles}__{control}")
    return os.path.join(os.path.join(out_dir, "smoke") if tiles == "smoke" else out_dir, tag + ".jsonl")


def gate_path(out_dir, config, model_tag, tier):
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", f"step4_gate__{config}__{model_tag}__{sampling_tag(tier)}")
    return os.path.join(out_dir, tag + ".json")


def make_record(side, res, model_tag, source):
    """One comp_run-style record from a sidecar row (labels, local) and an output row (answer, Kaggle)."""
    text = res["raw_response"]
    label, obj = cr.parse(text, side["config"], side["class_a_is"])
    _, had_trace = cr.strip_thinking(text)
    cells = sorted({c for e in ((obj or {}).get("evidence") or []) if isinstance(e, dict)
                    for c in (e.get("grid_cells") or []) if isinstance(c, str) and cr.CELL.match(c.strip().upper())}) \
        if isinstance(obj, dict) else []
    sampling = cr.TIERS[side["tier"]]
    served = f"{ij.PINNED_MODEL}@{res.get('model_revision')}" + (f"+lora:{res['adapter_sha256'][:12]}" if res.get("adapter_sha256") else "")
    return {"config": side["config"], "model": model_tag, "provider_pinned": PROVIDER, "provider_served": PROVIDER,
            "base_url": None, "served_model": served,
            "timings": {"seconds": res.get("seconds"), "batch_seconds": res.get("batch_seconds"), "batch_rows": res.get("batch_rows")},
            "finish_reason": res["finish_reason"], "tiles": side["tiles"], "control": side["control"],
            "image": side["image"], "image_shown": side["image_shown"],
            "label_true": side["label_true"], "ssa_votes": side["ssa_votes"], "class_a_is": side["class_a_is"],
            "fewshot": side["fewshot"], "prompt_sha256": side["prompt_sha256"],
            "temperature": sampling["temperature"], "sampling": sampling, "sampling_tag": sampling_tag(side["tier"]),
            "seed": side["seed"], "n_images": side["n_images"], "thinking_trace": had_trace,
            "n_valid_cited_cells": len(cells), "thinking_suppressed_token_id": ij.THINK_OPEN_ID,
            "endpoint": "transformers generate (processor.apply_chat_template)", "system_fingerprint": None,
            "raw_response": text, "label": label, "parsed_ok": obj is not None,
            "n_evidence": len(obj.get("evidence") or []) if isinstance(obj, dict) else None,
            "usage": {"prompt_tokens": res["n_prompt_tokens"], "completion_tokens": res["n_new_tokens"]},
            "cost_usd": 0, "attempts": [], "error": None, "extra_body": None,
            "kaggle": {"job_id": res["job_id"], "source_file": os.path.basename(source),
                       **{k: res.get(k) for k in ("model_revision", "weights_verified", "images_verified", "adapter_sha256",
                                                  "dtype", "quantization", "batch_size", "sampler", "run_signature",
                                                  "jobs_sha256", "prompt_ids_sha256", "max_new_tokens", "written_utc", "run")}}}


def load_summary(path):
    """train_summary.json of the training run the adapter comes from. Returns (summary, problems)."""
    try:
        with open(os.path.expanduser(path)) as fh:
            summary = json.load(fh)
    except (OSError, ValueError) as e:
        return None, [f"--train-summary {path}: cannot be read ({type(e).__name__})"]
    if not isinstance(summary, dict) or "status" not in summary or not isinstance(summary.get("best_adapter"), dict):
        return None, [f"--train-summary {path}: not a train_summary.json written by train_lora.py"]
    return summary, []


def provenance_problems(jid, res, smoke, args):
    """Is this answer from the model the command line says it is from? Fail-closed: what is not recorded as
    right counts as wrong."""
    out = []
    run = res.get("run") or {}
    trained = (run.get("adapter") or {}).get("trained") or {}
    sha = res.get("adapter_sha256")
    if args.base_model:
        if sha:
            out.append(f"{jid}: --base-model, but this answer was generated with an adapter (sha256 {str(sha)[:12]}...)")
        return out
    summary = args.summary
    want = summary["best_adapter"].get("adapter_model_sha256")
    if not sha:
        out.append(f"{jid}: --train-summary was given, but this answer was generated without an adapter (use --base-model)")
        return out
    if not want or sha != want:
        out.append(f"{jid}: the answer's adapter (sha256 {str(sha)[:12]}...) is not the best adapter of the training run in "
                   f"--train-summary (sha256 {str(want)[:12]}...)")
    if smoke:
        return out                       # pool tiles, never scored: any adapter of that run may be checked for format
    if summary.get("status") != "complete":
        out.append(f"{jid}: the training run is not finished (train_summary.json status {summary.get('status')!r}): its "
                   "best adapter so far is not the validation-selected one")
    if trained.get("final") is not True:
        out.append(f"{jid}: the adapter's step4_meta.json does not say final=true (final={trained.get('final')!r}): an epoch "
                   "checkpoint or the best-so-far adapter of an unfinished run cannot be scored on dev")
    if trained.get("epoch") != summary.get("chosen_epoch") or trained.get("epoch") is None:
        out.append(f"{jid}: the adapter is from epoch {trained.get('epoch')!r}, the training run chose epoch {summary.get('chosen_epoch')!r}")
    if run.get("allow_non_final_adapter"):
        out.append(f"{jid}: generated with infer_jobs.py --allow-non-final-adapter")
    if (trained.get("protocol_run") is not True or summary.get("protocol_run") is not True) and not args.allow_non_protocol_adapter:
        out.append(f"{jid}: the adapter does not come from a protocol run (step4_meta.json protocol_run={trained.get('protocol_run')!r}, "
                   f"train_summary.json protocol_run={summary.get('protocol_run')!r}): a debug or smoke run of train_lora.py")
    return out


def check_group(name, rows, local_dir, args, truth):
    """All checks for the answers of one job file. Returns (problems, sidecar rows in order, {job_id: (row, source)})."""
    problems = []
    side_path = os.path.join(local_dir, name + ".sidecar.jsonl")
    meta_path = os.path.join(local_dir, name + ".meta.json")
    if not os.path.exists(side_path) or not os.path.exists(meta_path):
        return [f"{name}: no sidecar/meta in {local_dir} (expected {os.path.basename(side_path)}); it is written by "
                "build_dev_jobs.py on this machine and is needed for the labels"], [], {}
    side_bytes = open(side_path, "rb").read()
    meta = json.load(open(meta_path))
    if hashlib.sha256(side_bytes).hexdigest() != meta.get("sidecar_sha256"):
        return [f"{name}: {os.path.basename(side_path)} is not the sidecar build_dev_jobs.py wrote with this meta file "
                "(sha256 differs): it was edited or comes from another build. Rebuild the job files; the labels are "
                "not taken from a sidecar that cannot be tied to the job file"], [], {}
    sidecar = [json.loads(l) for l in side_bytes.decode("utf-8").splitlines() if l.strip()]
    lab, votes, part = truth
    wrong = [s["image"] for s in sidecar if lab.get(s["image"]) != s["label_true"] or votes.get(s["image"]) != s["ssa_votes"]
             or part.get(s["image"]) != "train"]
    if wrong:
        return [f"{name}: {len(wrong)} sidecar rows disagree with annotations.csv (label, vote count or partition), "
                f"e.g. {wrong[:3]}"], [], {}
    side = {s["job_id"]: s for s in sidecar}
    counts = collections.Counter(r["job_id"] for r, _ in rows)
    dup = sorted(j for j, n in counts.items() if n > 1)
    if dup:
        problems.append(f"{name}: {len(dup)} job_ids appear more than once, e.g. {dup[:3]} (did you pass a merged file "
                        "together with its shard files?)")
    unknown = sorted(set(counts) - set(side))
    if unknown:
        problems.append(f"{name}: {len(unknown)} answers have no job in the sidecar, e.g. {unknown[:3]}")
    missing = [s["job_id"] for s in sidecar if s["job_id"] not in counts]
    if missing:
        problems.append(f"{name}: {len(missing)} of {len(sidecar)} jobs have no answer, e.g. {missing[:3]}; "
                        "re-run infer_jobs.py on Kaggle to finish the file")
    by_id = {}
    for res, source in rows:
        s = side.get(res["job_id"])
        if s is None or counts[res["job_id"]] > 1:
            continue
        by_id[res["job_id"]] = (res, source)
        jid = res["job_id"]
        lack = [k for k in ("raw_response", "n_prompt_tokens", "n_new_tokens", "finish_reason", "tier", "seed") if k not in res]
        if lack:
            problems.append(f"{jid}: output line lacks {lack}")
            continue
        for key, want in (("tier", s["tier"]), ("seed", s["seed"]), ("max_new_tokens", s["max_new_tokens"]), ("n_images", s["n_images"])):
            if key in res and res[key] != want:
                problems.append(f"{jid}: {key} is {res[key]!r} in the output but {want!r} in the sidecar")
        if not isinstance(res["raw_response"], str):
            problems.append(f"{jid}: raw_response is not a string")
        elif ij.THINK_OPEN in res["raw_response"]:
            problems.append(f"{jid}: the banned thinking opener {ij.THINK_OPEN} is in the answer")
        if res["finish_reason"] not in ("stop", "length"):
            problems.append(f"{jid}: finish_reason {res['finish_reason']!r}")
        if res["n_prompt_tokens"] < ij.IMAGE_TOKENS_PER_IMAGE * s["n_images"]:
            problems.append(f"{jid}: {res['n_prompt_tokens']} prompt tokens < 256 x {s['n_images']} images: image not ingested")
        if not args.skip_token_check and s.get("n_prompt_tokens_expected") is not None:
            if res["n_prompt_tokens"] != s["n_prompt_tokens_expected"] or res.get("prompt_ids_sha256") != s["prompt_ids_sha256_expected"]:
                problems.append(f"{jid}: the prompt was tokenized differently on Kaggle ({res['n_prompt_tokens']} tokens, hash "
                                f"{str(res.get('prompt_ids_sha256'))[:12]}) than by the official processor here "
                                f"({s['n_prompt_tokens_expected']} tokens, hash {s['prompt_ids_sha256_expected'][:12]})")
        if res.get("jobs_sha256") != meta["jobs_sha256"]:
            problems.append(f"{jid}: generated from another version of the job file (jobs_sha256 {str(res.get('jobs_sha256'))[:12]} "
                            f"vs {meta['jobs_sha256'][:12]} built here)")
        if res.get("weights_verified") is not True and not args.allow_unverified_weights:
            problems.append(f"{jid}: weights_verified is {res.get('weights_verified')!r} (not checked against the pinned revision)")
        if res.get("images_verified") is not True and not args.allow_unverified_images:
            problems.append(f"{jid}: images_verified is {res.get('images_verified')!r}: the image bytes were not checked against the "
                            "job file's sha256 on Kaggle (the prompt token hash covers token ids, not pixels)")
        problems += provenance_problems(jid, res, s["tiles"] == "smoke", args)
    sigs = collections.Counter(r.get("run_signature") for r, _ in by_id.values())
    if len(sigs) > 1 and not args.allow_mixed_runs:
        problems.append(f"{name}: the answers come from {len(sigs)} different run settings {dict(sigs)}; "
                        "pass --allow-mixed-runs only if that is intended and documented")
    # collapse repeated per-job messages of the same kind
    if len(problems) > 12:
        kinds = collections.Counter(re.sub(r"^[^:]+: ", "", p)[:60] for p in problems)
        problems = problems[:12] + [f"{name}: {n} problems like: {k} ..." for k, n in kinds.most_common(5)]
    return problems, sidecar, by_id


def format_rate(recs):
    """Share of ALL records with a valid label and at least one valid cited grid cell (the addendum's format rule)."""
    return sum(r["label"] in ("HP", "SSA") and (r["n_valid_cited_cells"] or 0) >= 1 for r in recs) / len(recs)


def smoke_check(recs):
    """The label-free check before dev (PLAN, step-4 Amendment 2). An answer counts only if it ends with a normal
    stop AND has a valid label AND at least one valid cited grid cell. -> (n_good, n_needed, ok): ok when at
    least 90% of the answers count (18 of 20). A cut-off answer never counts, whatever could be read from it."""
    good = sum(r["finish_reason"] == "stop" and r["label"] in ("HP", "SSA") and (r["n_valid_cited_cells"] or 0) >= 1 for r in recs)
    need = -(-9 * len(recs) // 10)                      # ceil(0.9 n) in integers: 18 of 20
    assert FORMAT_BAR == 0.90
    return good, need, good >= need


def health(recs):
    """Label-free summary of a set of records (scoring is comp_analyze.py's job)."""
    n = len(recs)
    usable = [r for r in recs if r["label"] in ("HP", "SSA")]
    return (f"n {n}; finish stop {sum(r['finish_reason'] == 'stop' for r in recs)} / length "
            f"{sum(r['finish_reason'] == 'length' for r in recs)}; valid label {len(usable) / n:.1%}; "
            f"valid label and >= 1 valid grid cell {format_rate(recs):.1%}; thinking trace {sum(r['thinking_trace'] for r in recs) / n:.1%}; "
            f"mean new tokens {sum(r['usage']['completion_tokens'] for r in recs) / n:.0f}; "
            f"mean s/job {sum((r['timings']['seconds'] or 0) for r in recs) / n:.1f}")


def read_ledger(out_dir):
    path = os.path.join(out_dir, LEDGER)
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True, help="output .jsonl file(s) downloaded from /kaggle/working/")
    ap.add_argument("--model-tag", required=True,
                    help="model name for the records and file names, e.g. medgemma-1.5-4b-it-lora-r16 "
                         "(letters, digits, dot, dash, underscore; no double underscore)")
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--train-summary", metavar="PATH",
                     help="train_summary.json of the finished train_lora.py run whose final adapter generated the answers")
    who.add_argument("--base-model", action="store_true", help="the answers come from the base model, no adapter")
    ap.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR,
                    help="folder with the sidecars written by build_dev_jobs.py (default: %(default)s)")
    ap.add_argument("--out-dir", default=cr.OUT, help="where the record files go (default: %(default)s)")
    ap.add_argument("--control-tiles", default="split", choices=["split", "dev"],
                    help="image controls only: split = screen + dev_rest files like the main run (default); dev = one "
                         "300-tile file with tiles='dev' in comp_run's order, as run_medgemma_search.sh produces")
    ap.add_argument("--force", action="store_true", help="overwrite existing record files of the same run")
    ap.add_argument("--allow-mixed-runs", action="store_true", help="accept answers from more than one run setting")
    ap.add_argument("--allow-unverified-weights", action="store_true",
                    help="accept answers whose weights were not hashed against the pinned revision")
    ap.add_argument("--allow-unverified-images", action="store_true",
                    help="accept answers generated with infer_jobs.py --no-verify-images")
    ap.add_argument("--allow-non-protocol-adapter", action="store_true",
                    help="accept dev answers from the final adapter of a training run marked protocol_run=false "
                         "(exploratory only; recorded in the ledger). A non-final adapter is never accepted")
    ap.add_argument("--allow-another-adapter", action="store_true",
                    help="accept dev answers although another adapter was already imported for the same configuration, "
                         "tier and control (recorded in the ledger; dev is meant to be evaluated once per tier)")
    ap.add_argument("--skip-token-check", action="store_true",
                    help="do not compare the Kaggle prompt token count / hash with the local expectation")
    ap.add_argument("--dry-run", action="store_true", help="run every check, print the summary, write nothing")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.model_tag) or "__" in args.model_tag:
        ap.error("--model-tag may contain letters, digits, dot, dash and single underscores only")
    args.summary = None
    if args.train_summary:
        args.summary, problems = load_summary(args.train_summary)
        if problems:
            refuse(problems)

    groups = collections.OrderedDict()
    for path in args.results:
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            refuse([f"{path} does not exist"])
        raw = open(path, "rb").read()
        if raw and not raw.endswith(b"\n"):
            refuse([f"{path} ends in the middle of a line: the download or the run was cut off"])
        for res in ij.read_output(path):
            groups.setdefault(str(res["job_id"]).rsplit("__", 1)[0], []).append((res, path))
    if not groups:
        refuse(["no answers in the given files"])

    truth = cr.labels()
    ledger = read_ledger(args.out_dir)
    problems, plans, ctx_cache = [], [], {}
    for name, rows in groups.items():
        p, sidecar, by_id = check_group(name, rows, os.path.abspath(os.path.expanduser(args.local_dir)), args, truth)
        problems += p
        if p:
            continue
        recs = [make_record(s, by_id[s["job_id"]][0], args.model_tag, by_id[s["job_id"]][1]) for s in sidecar]
        s0 = sidecar[0]
        by_tiles = collections.OrderedDict()
        for r in recs:
            by_tiles.setdefault(r["tiles"], []).append(r)
        if s0["tiles"] != "smoke":
            sizes = {t: len(v) for t, v in by_tiles.items()}
            if sizes != {"screen": 100, "dev_rest": 200}:
                problems.append(f"{name}: expected 100 screen + 200 dev_rest tiles, got {sizes}")
            elif s0["control"] != "none" and args.control_tiles == "dev":
                import build_dev_jobs as bj
                ctx = ctx_cache.setdefault("ctx", bj.load_context())
                rank = {t: n for n, t in enumerate(bj.comp_run_order(ctx["splits"]["dev"], ctx, s0["config"]))}
                by_tiles = collections.OrderedDict(dev=sorted(({**r, "tiles": "dev"} for r in recs), key=lambda r: rank[r["image"]]))
            sha = recs[0]["kaggle"]["adapter_sha256"]
            others = sorted({str(e.get("adapter_sha256"))[:12] for e in ledger
                             if (e.get("config"), e.get("tier"), e.get("control")) == (s0["config"], s0["tier"], s0["control"])
                             and e.get("adapter_sha256") and sha and e["adapter_sha256"] != sha})
            if others and not args.allow_another_adapter:
                problems.append(f"{name}: the ledger ({LEDGER}) already holds a dev import for {s0['config']}, tier {s0['tier']}, "
                                f"control {s0['control']} with another adapter ({', '.join(others)}...); this one is {sha[:12]}.... "
                                "Evaluating several adapters on dev and keeping one is checkpoint selection on dev; "
                                "--allow-another-adapter overrides and is recorded")
        targets = {t: record_path(args.out_dir, s0["config"], args.model_tag, s0["tier"], t, s0["control"]) for t in by_tiles}
        clash = [record_path(args.out_dir, s0["config"], args.model_tag, s0["tier"], t, s0["control"])
                 for t in (("smoke",) if s0["tiles"] == "smoke" else ("screen", "dev_rest", "dev"))]
        clash = [c for c in clash if os.path.exists(c)]
        if clash and not args.force:
            problems.append(f"{name}: record file(s) for this configuration, model tag, tier and control already exist: "
                            f"{[os.path.basename(c) for c in clash]}; use another --model-tag, or --force to replace them")
        plans.append((name, recs, by_tiles, targets, clash, s0["tier"]))
    if problems:
        refuse(problems)

    smoke_failed = False
    for name, recs, by_tiles, targets, clash, tier in plans:
        run = recs[0]["kaggle"]
        r0 = recs[0]
        smoke = r0["tiles"] == "smoke"
        print(f"{name}: {health(recs)}")
        as_trained = (run.get("run") or {}).get("precision_as_trained")
        if as_trained is False:
            print("  WARNING: the base model was NOT loaded with the precision the adapter was trained with (see kaggle.run.precision)")
        print(f"  model {recs[0]['served_model']}, dtype {run['dtype']}"
              f"{', ' + run['quantization'] if run.get('quantization') else ''}, batch size {run['batch_size']}, "
              f"sampler {run['sampler']}, weights verified {run['weights_verified']}, images verified {run['images_verified']}, GPUs "
              f"{sorted({g for r in recs for g in ((r['kaggle'].get('run') or {}).get('gpus') or [])})}")
        fmt = format_rate(recs)
        gate = None
        if smoke and r0["control"] == "none":
            stops = sum(r["finish_reason"] == "stop" for r in recs)
            good, need, ok = smoke_check(recs)
            smoke_failed = smoke_failed or not ok
            print(f"  SMOKE FORMAT CHECK (label-free, pool tiles; >= {FORMAT_BAR:.0%} of answers must end with a normal stop AND have a "
                  f"valid label + grid cell): {good}/{len(recs)} such answers, {need} needed (stop {stops}/{len(recs)}, valid label + "
                  f"grid cell {fmt:.1%}) -> " + ("ok" if ok else "FAILS: do NOT run the dev jobs with this adapter"))
        elif not smoke and r0["control"] == "none":
            ok = fmt >= FORMAT_BAR
            print(f"  STEP-4 FORMAT GATE (>= {FORMAT_BAR:.0%} of all {len(recs)} dev answers with a valid label and >= 1 valid grid cell): "
                  f"{fmt:.1%} -> " + ("ok" if ok else "FAILS (the configuration does not pass dev whatever comp_analyze prints)"))
            gate = {"config": r0["config"], "model_tag": args.model_tag, "sampling_tag": r0["sampling_tag"], "control": "none",
                    "n": len(recs), "n_valid_label_and_cell": round(fmt * len(recs)), "format_rate": round(fmt, 4),
                    "format_bar": FORMAT_BAR, "format_ok": ok, "adapter_sha256": run["adapter_sha256"],
                    "rule": "PLAN step-4 addendum: at least 90% of dev answers parse with a valid label and at least one "
                            "valid cited grid cell. Step-4 dev pass = comp_analyze dev PASS and format_ok and both image controls OK.",
                    "written": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        if args.dry_run:
            print("  dry run: " + ", ".join(os.path.basename(t) for t in targets.values()) + " NOT written")
            continue
        for c in clash:                                   # --force: replace the whole earlier import of this run
            os.remove(c)
        for tiles, path in targets.items():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                for r in by_tiles[tiles]:
                    fh.write(json.dumps(r) + "\n")
            os.replace(tmp, path)
            print(f"  wrote {len(by_tiles[tiles])} records -> {path}")
        if gate is not None:
            path = gate_path(args.out_dir, r0["config"], args.model_tag, tier)
            with open(path + ".tmp", "w") as fh:
                json.dump(gate, fh, indent=1)
                fh.write("\n")
            os.replace(path + ".tmp", path)
            print(f"  wrote the format gate -> {path}")
        if not smoke:
            entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "name": name, "config": r0["config"],
                     "tier": tier, "control": r0["control"],
                     "model_tag": args.model_tag, "adapter_sha256": run["adapter_sha256"], "base_model": bool(args.base_model),
                     "run_signatures": sorted({str(r["kaggle"]["run_signature"]) for r in recs}),
                     "jobs_sha256": run["jobs_sha256"], "n": len(recs), "format_rate": round(fmt, 4),
                     "source_files": sorted({r["kaggle"]["source_file"] for r in recs}),
                     "train_summary": None if args.summary is None else {
                         "status": args.summary.get("status"), "protocol_run": args.summary.get("protocol_run"),
                         "chosen_epoch": args.summary.get("chosen_epoch"), "fingerprint": args.summary.get("fingerprint")},
                     "overrides": sorted(k for k in ("force", "allow_mixed_runs", "allow_unverified_weights", "allow_unverified_images",
                                                     "allow_non_protocol_adapter", "allow_another_adapter", "skip_token_check")
                                         if getattr(args, k))}
            os.makedirs(args.out_dir, exist_ok=True)
            with open(os.path.join(args.out_dir, LEDGER), "a") as fh:
                fh.write(json.dumps(entry, sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
    if not args.dry_run:
        print("next: python3 comp_analyze.py   (scores every file in runs/competence/ with the PLAN's accuracy bars; the format "
              "gate is the step4_gate__*.json file written here, and both must hold)")
    if smoke_failed:
        print("SMOKE FORMAT CHECK FAILED: the adapter does not write the full cte_p1 answer reliably. Do not run the dev jobs.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
