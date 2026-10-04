#!/usr/bin/env python3
r"""CPU-only tests for the Kaggle batch-inference glue. No model weights are loaded and no network is used.

    ~/mhist_local/venv/bin/python kaggle_ft/test_jobs_local.py            # everything (about 5 minutes)
    ~/mhist_local/venv/bin/python kaggle_ft/test_jobs_local.py --only tokens leak
    ~/mhist_local/venv/bin/python kaggle_ft/test_jobs_local.py --no-tiny-model

Needs the local venv (transformers + torch, CPU) and the official processor files in ~/mhist_local/medgemma-1.5-4b-it
(or MEDGEMMA_DIR). Everything it writes goes to a folder of its own under ~/mhist_local/tmp/ (MHIST_TEST_TMP names
another base) and is removed at the end (--keep keeps it). Nothing is written inside the project tree: the temp
folder holds label sidecars and links to the model's tokenizer files, and the project tree is copied to a public
repository. Each run has its own folder, so two runs at the same time do not disturb each other.

What is tested

  reference   comp_run.main() itself is run for every config x control x tier against a fake in-process HTTP
              session (no socket is opened; the real server on port 8089 is never touched) with its output
              folder redirected to the temp dir. That captures the exact prompt string, images, seed and sampling
              parameters comp_run sends for each of the 300 dev tiles, and the records it writes.
  jobs        build_dev_jobs.py's jobs equal that capture for every tile: prompt string (comp_run's HF-identical
              string logic applied to the job's parts), image bytes, seed, tier parameters, token limit; and the
              sidecar equals comp_run's record fields (label_true, ssa_votes, class_a_is, image_shown, fewshot,
              prompt_sha256, n_images, tiles).
  tokens      for each config x control, apply_chat_template on a job's parts (infer_jobs.encode_job, the real
              processor) gives the token ids of comp_run's prompt string with every image marker replaced by
              <start_of_image> + 256 image tokens + <end_of_image>; cte_p1 with one image is 822 tokens.
  leak        job files hold exactly the five job keys; they are invariant to a permutation of the dev labels;
              no text signature determines a label; build_bundle.audit_dir passes on the jobs folder and trips
              on the sidecar folder; only train-partition tiles are referenced.
  import      a fabricated Kaggle output file round-trips through import_results.py into records that agree
              with comp_run's own records field by field, that comp_analyze.summarize scores identically, and
              that comp_analyze.main() groups into screen + 300-tile dev rows; every refusal fires (adapter
              provenance, sidecar integrity, image verification, the one-adapter ledger); the step-4 format
              gate and the label-free smoke check are computed and written.
  pinned      the hash constants in infer_jobs.py equal the official files' hashes (small files hashed here,
              the two weight files compared with the download metadata; the weights are not read).
  sampler     the per-job sampler draws from the distribution stock HF sampling uses (same kept tokens and
              probabilities as transformers' own warpers, empirical frequencies), never draws a banned token,
              and a row's draws depend on its seed only, not on the batch around it.
  tiny        infer_jobs.py end to end on CPU with a tiny randomly initialised Gemma3 model and the real
              processor: output fields, weight-check refusal, resume after a cut-off line, order and worker
              invariance, forced end-of-turn handling, the thinking-token ban, --sampler hf, and a batch-size
              1 vs 3 comparison (reported, not asserted: float-level batch dependence is expected and documented);
              the adapter gate for dev jobs, the 4-bit scope check and skip list, token_type_ids, images_verified.
"""

import argparse
import base64
import contextlib
import functools
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time
import traceback
import types

sys.dont_write_bytecode = True                  # no __pycache__ next to the project's modules
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FAKE_BASE = "http://127.0.0.1:9/v1"            # comp_run's LOCAL mode; port 9 is never contacted (fake session)
os.environ["COMP_BASE_URL"] = FAKE_BASE
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import comp_analyze as ca      # noqa: E402
import comp_run as cr          # noqa: E402
import build_dev_jobs as bj    # noqa: E402
import import_results as ir    # noqa: E402
import infer_jobs as ij        # noqa: E402

assert cr.LOCAL and cr.BASE_URL == FAKE_BASE and "8089" not in cr.BASE_URL
REAL_OUT = cr.OUT
TMP_BASE = os.path.abspath(os.path.expanduser(os.environ.get("MHIST_TEST_TMP") or "~/mhist_local/tmp"))
TMP = None                                     # this run's own folder under TMP_BASE, made by setup()
MARKER = "<__media__>"
FAKE_ADAPTER_SHA = "ab" * 32                   # the "final adapter" of the fabricated training run
MODEL_DIR = bj.DEFAULT_PROCESSOR_DIR
CONFIGS, CONTROLS, TIERS = bj.CONFIGS, bj.CONTROLS, bj.TIERS
S = {}                                         # shared state between tests


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


@contextlib.contextmanager
def quiet(stdout=True, stderr=True):
    with open(os.devnull, "w") as null, contextlib.ExitStack() as stack:
        if stdout:
            stack.enter_context(contextlib.redirect_stdout(null))
        if stderr:
            stack.enter_context(contextlib.redirect_stderr(null))
        yield


@contextlib.contextmanager
def quiet_fds():
    """Like quiet(), but at file-descriptor level, so that child processes are silenced too."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1)
        os.dup2(null, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in saved + [null]:
            os.close(fd)


def exits(fn, *a, **k):
    """Run fn; return its SystemExit code (None if it returned normally)."""
    try:
        with quiet():
            fn(*a, **k)
    except SystemExit as e:
        return e.code if e.code is not None else 0
    return None


def main_code(fn, argv):
    """Exit code of a command-line entry point, whether it returns it or raises SystemExit."""
    try:
        with quiet(), quiet_fds():
            code = fn(argv)
    except SystemExit as e:
        code = e.code
    return 0 if code is None else code


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def tree_state(path):
    return sorted((os.path.relpath(os.path.join(b, f), path), os.path.getsize(os.path.join(b, f)))
                  for b, _, fs in os.walk(path) for f in fs) if os.path.isdir(path) else []


# ------------------------------------------------------------------------------------- fabricated answers

def fabricate(img, config, a_is, true_label):
    """A deterministic fake model answer per tile, covering every parse path. Returns (text, finish_reason)."""
    h = int(hashlib.sha256(f"fab:{config}:{img}".encode()).hexdigest()[:8], 16)
    kind = h % 20
    pred = true_label if (h // 20) % 4 else ("SSA" if true_label == "HP" else "HP")       # about 75% right
    if config.startswith("neutral"):
        token = "A" if pred == a_is else "B"
        if (h // 80) % 3 == 0:
            token = "Class " + token
    else:
        token = pred
    cells = [["B2"], ["A1", "c3 "], ["D4", "B2", "A4"], ["A1", "A2", "B1", "B2"]][(h // 240) % 4]
    ev = [] if config == "co_p1" else [{"feature": "serration", "grid_cells": cells, "description": "x"}]
    body = json.dumps({"label": token, "confidence": 0.8, "evidence": ev}, indent=2)
    text, finish = body, "stop"
    if kind == 0:
        text = "I cannot classify this tile."
    elif kind == 1:
        text, finish = body[: len(body) // 2], "length"
    elif kind == 2:
        text = "thought\nlooking at the crypts" + ij.THINK_CLOSE + body
    elif kind == 3:
        text = "```json\n" + body + "\n```"
    elif kind == 4:
        text = body.replace(json.dumps(token), '"maybe"', 1)
    elif kind == 5:
        text = json.dumps({"label": token, "confidence": 0.5,
                           "evidence": [{"feature": "stroma", "grid_cells": ["Z9", 7], "description": "y"}]})
    if kind % 2 == 0 and finish == "stop":
        text += "<end_of_turn>"
    return text, finish


# ------------------------------------------------------------------------------- comp_run reference runs

class FakeResponse:
    def __init__(self, payload):
        self.status_code, self._payload, self.text = 200, payload, json.dumps(payload)[:200]

    def json(self):
        return self._payload


class FakeSession:
    """Stands in for requests.Session inside comp_run: answers /props, /tokenize and /completion in process."""
    calls, answers = [], []

    def get(self, url, timeout=None):
        check(url == FAKE_BASE.rsplit("/v1", 1)[0] + "/props", f"unexpected GET {url}")
        return FakeResponse({"media_marker": MARKER})

    def post(self, url, headers=None, json=None, timeout=None):       # noqa: A002 - mirrors requests' signature
        root = FAKE_BASE.rsplit("/v1", 1)[0]
        if url == root + "/tokenize":
            return FakeResponse({"tokens": [ij.THINK_OPEN_ID]})
        check(url == root + "/completion", f"unexpected POST {url}")
        text, finish = FakeSession.answers[json["seed"]]        # the per-tile seed identifies the tile of this call
        FakeSession.calls.append(json)
        return FakeResponse({"content": text, "stop_type": {"stop": "eos", "length": "limit"}[finish],
                             "tokens_evaluated": 999999, "tokens_predicted": 1, "timings": {}, "model": "fake"})


def run_comp_run(config, tiles, control, tier):
    """Run the real comp_run.main() for one file. Returns [(request body, record)] in comp_run's run order."""
    lab = S["ctx"]["lab"]
    order = bj.tile_list("smoke", S["ctx"])[0] if tiles == "smoke" else S["ctx"]["splits"][tiles]
    FakeSession.calls = []
    FakeSession.answers = {bj.job_seed(config, control, tier, img): fabricate(img, config, S["ctx"]["ab"][img], lab[img])
                           for img in order}
    check(len(FakeSession.answers) == len(order), "seed collision between tiles")
    argv = sys.argv
    sys.argv = ["comp_run.py", "--model", "fake-model", "--provider", "local", "--config", config,
                "--tiles", tiles, "--control", control, "--tier", tier]
    try:
        with quiet():
            cr.main()
    finally:
        sys.argv = argv
    path = os.path.join(cr.OUT, "smoke" if tiles == "smoke" else "",
                        f"{config}__fake-model__local-{ir.sampling_tag(tier)}__{tiles}__{control}.jsonl")
    recs = read_jsonl(path)
    check(len(recs) == len(order) == len(FakeSession.calls), f"{path}: {len(recs)} records for {len(order)} tiles")
    check(sorted(r["image"] for r in recs) == sorted(order), "comp_run did not process exactly the expected tiles")
    for body, rec in zip(FakeSession.calls, recs):          # each call got the answer fabricated for ITS tile
        check(rec["raw_response"] == fabricate(rec["image"], config, S["ctx"]["ab"][rec["image"]], lab[rec["image"]])[0]
              and rec["seed"] == body["seed"], f"{path}: call and record are not aligned for {rec['image']}")
    return list(zip(FakeSession.calls, recs))


def write_summary(path, **override):
    """A train_summary.json as train_lora.py writes it for a finished protocol run (the fields import_results reads)."""
    summary = {"status": "complete", "protocol_run": True, "chosen_epoch": 3, "epochs_completed": 4, "epochs_planned": 4,
               "fingerprint": "f" * 64, "best_adapter": {"dir": "best_adapter", "adapter_model_sha256": FAKE_ADAPTER_SHA, "final": True}}
    summary.update(override)
    with open(path, "w") as fh:
        json.dump(summary, fh)
    return path


def setup():
    global TMP
    if bj.in_synced_folder(TMP_BASE):
        sys.exit(f"STOP: the test folder base {TMP_BASE} is inside an iCloud-synced folder or the project tree; "
                 "set MHIST_TEST_TMP to a folder outside them (default: ~/mhist_local/tmp)")
    os.makedirs(TMP_BASE, exist_ok=True)
    TMP = tempfile.mkdtemp(prefix="mhist_test_jobs_", dir=TMP_BASE)
    comp = os.path.join(TMP, "competence")
    os.makedirs(comp)
    for f in ("splits.json", "ab_assignment.json", "fewshot_examples.json"):
        shutil.copy(os.path.join(REAL_OUT, f), os.path.join(comp, f))
    cr.OUT = comp                                               # comp_run now reads and writes ONLY here
    cr.requests = types.SimpleNamespace(Session=FakeSession, RequestException=cr.requests.RequestException)
    cr.b64_grid = functools.lru_cache(maxsize=None)(cr.b64_grid)   # same bytes, rendered once per tile
    S["ctx"] = bj.load_context()
    S["dirs"] = {k: os.path.join(TMP, k) for k in ("jobs", "local", "bundle", "records")}
    S["summary"] = write_summary(os.path.join(TMP, "train_summary.json"))


def t_reference():
    """Capture what comp_run.main() sends and records, for every config x control x tier."""
    ref, n_calls = {}, 0
    for config in CONFIGS:
        for tier in TIERS:
            pairs = run_comp_run(config, "screen", "none", tier) + run_comp_run(config, "dev_rest", "none", tier)
            ref[(config, "none", tier)] = {rec["image"]: (body, rec) for body, rec in pairs}
            for control in ("noimage", "mismatch"):
                ref[(config, control, tier)] = {rec["image"]: (body, rec) for body, rec in run_comp_run(config, "dev", control, tier)}
            n_calls += 900
    check(all(len(v) == 300 for v in ref.values()), "a reference run does not cover 300 tiles")
    S["ref"] = ref
    S["ref_smoke"] = {(config, control): {rec["image"]: (body, rec) for body, rec in run_comp_run(config, "smoke", control, "1")}
                      for config in ("cte_p1", "neutral_crit_fs") for control in CONTROLS}
    return f"{len(ref)} dev + {len(S['ref_smoke'])} smoke comp_run runs captured ({n_calls + 120} calls), fake session only"


def t_build():
    """Build every job file through the command-line entry point."""
    d = S["dirs"]
    # step order: without the statement that step 3 is closed, no dev job file is built (smoke files are)
    code = exits(bj.main, ["--config", "cte_p1", "--jobs-dir", d["jobs"], "--local-dir", d["local"], "--bundle-dir", d["bundle"]])
    check(code not in (None, 0) and "step 3" in str(code) and not os.path.exists(d["jobs"]) and not os.path.exists(d["local"]),
          f"dev job files were built without --step3-closed: {str(code)[:80]}")
    code = exits(bj.main, ["--config", "cte_p1", "--step3-closed", "--jobs-dir", d["jobs"], "--bundle-dir", d["bundle"],
                           "--local-dir", os.path.join(ROOT, "runs", "x_local_only")])
    check(code not in (None, 0) and "public repository" in str(code) and not os.path.exists(os.path.join(ROOT, "runs", "x_local_only")),
          "a sidecar folder inside the project tree was accepted")
    with quiet(stderr=False):
        bj.main(["--config", *CONFIGS, "--control", *CONTROLS, "--tier", *TIERS, "--jobs-dir", d["jobs"], "--step3-closed",
                 "--local-dir", d["local"], "--bundle-dir", d["bundle"], "--write-images", "--expect-tokens", "require"])
        bj.main(["--config", "cte_p1", "neutral_crit_fs", "--control", *CONTROLS, "--tiles", "smoke", "--jobs-dir", d["jobs"],
                 "--local-dir", d["local"], "--bundle-dir", d["bundle"], "--write-images", "--expect-tokens", "require"])
    jobs, side = {}, {}
    for config in CONFIGS:
        for control in CONTROLS:
            for tier in TIERS:
                name = bj.job_name("dev", config, control, tier)
                jobs[(config, control, tier)], sha = ij.read_jobs(os.path.join(d["jobs"], name + ".jsonl"))
                side[(config, control, tier)] = read_jsonl(os.path.join(d["local"], name + ".sidecar.jsonl"))
                meta = json.load(open(os.path.join(d["local"], name + ".meta.json")))
                check(meta["jobs_sha256"] == sha and meta["n_jobs"] == 300, f"{name}: meta does not describe the job file")
    S["jobs"], S["side"] = jobs, side
    before = tree_state(d["jobs"])
    with quiet():                                               # idempotent: a second build changes nothing
        bj.main(["--config", "cte_p1", "--control", "none", "--tier", "1", "--jobs-dir", d["jobs"], "--step3-closed",
                 "--local-dir", d["local"], "--bundle-dir", d["bundle"], "--expect-tokens", "require"])
    check(tree_state(d["jobs"]) == before, "rebuilding changed the jobs folder")
    path = os.path.join(d["jobs"], "dev__cte_p1__none__tier1.jsonl")
    original = open(path).read()
    open(path, "w").write(original.replace('"seed": ', '"seed": 1', 1))
    code = exits(bj.main, ["--config", "cte_p1", "--step3-closed", "--jobs-dir", d["jobs"], "--local-dir", d["local"], "--bundle-dir", d["bundle"]])
    open(path, "w").write(original)
    check(code not in (None, 0), "a job file that differs was overwritten without --force")
    code = exits(bj.main, ["--config", "cte_p1", "--step3-closed", "--jobs-dir", d["jobs"], "--local-dir", os.path.join(d["jobs"], "x")])
    check(code not in (None, 0), "a sidecar folder inside the upload folder was accepted")
    side_path = os.path.join(d["local"], "dev__cte_p1__none__tier1.sidecar.jsonl")
    before_side = open(side_path).read()
    check(exits(bj.main, ["--config", "cte_p1", "--step3-closed", "--jobs-dir", d["jobs"], "--local-dir", d["local"], "--bundle-dir", d["bundle"],
                          "--expect-tokens", "off"]) is None and open(side_path).read() == before_side,
          "a build without the tokenizer stripped the token expectations from the sidecar")
    check(not bj.in_synced_folder(bj.DEFAULT_JOBS_DIR) and not bj.in_synced_folder(bj.DEFAULT_LOCAL_DIR)
          and ir.DEFAULT_LOCAL_DIR == bj.DEFAULT_LOCAL_DIR, "the default jobs / sidecar folders are not outside the synced tree")
    return (f"{len(jobs)} dev job files x 300 jobs + 6 smoke files; dev files need --step3-closed; rebuild is a no-op; a changed "
            "file, a nested sidecar folder and a sidecar folder inside the project tree are refused")


def llama_prompt_string(parts):
    """comp_run.py's HF-identical prompt string (PLAN amendment 2), applied to a job's parts."""
    return "<start_of_turn>user\n" + "".join(
        p["text"].strip() if p["type"] == "text" else "\n\n" + MARKER + "\n\n" for p in parts
    ) + "<end_of_turn>\n<start_of_turn>model\n"


@functools.lru_cache(maxsize=None)
def bundle_b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()


def compare_job(job, s, body, rec, tier, control, d):
    """One job + sidecar row against the request comp_run sent and the record it wrote for the same tile."""
    prompt = body["prompt"]
    ps, imgs = (prompt["prompt_string"], prompt["multimodal_data"]) if isinstance(prompt, dict) else (prompt, [])
    check(llama_prompt_string(job["parts"]) == ps, f"{job['job_id']}: prompt string differs from comp_run's")
    image_parts = [p for p in job["parts"] if p["type"] == "image"]
    mine = [bundle_b64(os.path.join(d["bundle"], p["file"])) for p in image_parts]
    check(mine == imgs, f"{job['job_id']}: image bytes or order differ from comp_run's")
    check(all(hashlib.sha256(base64.b64decode(b)).hexdigest() == p["sha256"] for b, p in zip(imgs, image_parts)), "image sha256 wrong")
    check(job["seed"] == body["seed"] == rec["seed"] == s["seed"], f"{job['job_id']}: seed differs")
    check(job["max_new_tokens"] == body["n_predict"] == 1500, "token limit differs")
    check(job["tier"] == tier and body["temperature"] == (1.0 if tier == "1" else 0.0)
          and body["top_k"] == 64 and body["top_p"] == 0.95, "tier parameters differ")
    check(body["logit_bias"] == [[ij.THINK_OPEN_ID, False]], "comp_run does not ban token 100 here")
    if tier == "1":
        t = ij.TIERS["1"]
        check((t["temperature"], t["top_k"], t["top_p"]) == (body["temperature"], body["top_k"], body["top_p"]),
              "infer_jobs tier-1 parameters differ from comp_run's")
    for f in ("image", "image_shown", "label_true", "ssa_votes", "class_a_is", "fewshot", "prompt_sha256",
              "n_images", "config", "control"):
        check(s[f] == rec[f], f"{job['job_id']}: sidecar {f} = {s[f]!r}, comp_run recorded {rec[f]!r}")


def t_jobs():
    """Jobs and sidecar equal the captured comp_run run, tile by tile."""
    d, n = S["dirs"], 0
    for key, by_img in S["ref"].items():
        config, control, tier = key
        check(len(S["jobs"][key]) == 300, f"{key}: {len(S['jobs'][key])} jobs")
        for job, s in zip(S["jobs"][key], S["side"][key]):
            check(job["job_id"] == s["job_id"], "job and sidecar order differ")
            body, rec = by_img[s["image"]]
            compare_job(job, s, body, rec, tier, control, d)
            if control == "none":
                check(s["tiles"] == rec["tiles"], f"{job['job_id']}: tiles {s['tiles']} vs {rec['tiles']}")
            n += 1
    screen = set(S["ctx"]["splits"]["screen"])
    check(all((s["tiles"] == "screen") == (s["image"] in screen) for v in S["side"].values() for s in v), "tiles field wrong")
    mm = S["side"][("cte_p1", "mismatch", "1")]
    check(all(s["image_shown"] != s["image"] for s in mm) and sorted(s["image_shown"] for s in mm) == sorted(s["image"] for s in mm),
          "mismatch control is not a derangement of the dev tiles")
    n_smoke = 0
    for (config, control), by_img in S["ref_smoke"].items():
        name = bj.job_name("smoke", config, control, "1")
        jobs, _ = ij.read_jobs(os.path.join(d["jobs"], name + ".jsonl"))
        side = read_jsonl(os.path.join(d["local"], name + ".sidecar.jsonl"))
        check(len(jobs) == len(by_img) == 20, f"{name}: {len(jobs)} smoke jobs")
        for job, s in zip(jobs, side):
            body, rec = by_img[s["image"]]
            compare_job(job, s, body, rec, "1", control, d)
            check(s["tiles"] == rec["tiles"] == "smoke", "smoke tiles field")
            n_smoke += 1
    return (f"{n} dev jobs and {n_smoke} smoke jobs equal comp_run's requests and records (prompt string, image bytes, "
            "seed, sampling; labels only in the sidecar)")


def t_tokens():
    """apply_chat_template on a job's parts gives the token ids of comp_run's prompt string."""
    from transformers import AutoProcessor
    proc = S.get("processor") or AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    S["processor"] = proc
    tok, ids = proc.tokenizer, ij.special_ids(proc)
    boi, eoi = tok.convert_tokens_to_ids("<start_of_image>"), tok.convert_tokens_to_ids("<end_of_image>")
    check((boi, eoi, ids["image"], tok.bos_token_id) == (255999, 256000, 262144, 2), "special token ids differ from the model card")
    n, lengths = 0, {}
    for config in CONFIGS:
        for control in CONTROLS:
            key = (config, control, "1")
            picks, seen = [], {}
            for job, s in zip(S["jobs"][key], S["side"][key]):       # two tiles per Class-A assignment
                a = S["ctx"]["ab"][s["image"]]
                if seen.get(a, 0) < 2:
                    seen[a] = seen.get(a, 0) + 1
                    picks.append((job, s))
            for job, s in picks:
                body, _ = S["ref"][key][s["image"]]
                ps = body["prompt"]["prompt_string"] if isinstance(body["prompt"], dict) else body["prompt"]
                want = [tok.bos_token_id]
                for i, chunk in enumerate(ps.split(MARKER)):
                    if i:
                        want += [boi] + [ids["image"]] * 256 + [eoi]
                    want += tok(chunk, add_special_tokens=False)["input_ids"]
                enc = ij.encode_job(proc, job, S["dirs"]["bundle"], ids)
                got = enc["input_ids"][0].tolist()
                check(got == want, f"{job['job_id']}: token ids differ from comp_run's prompt ({len(got)} vs {len(want)} tokens)")
                check(got == tok(bj.hf_prompt_text(job["parts"]), add_special_tokens=False)["input_ids"],
                      f"{job['job_id']}: builder's expected ids differ from the processor's")
                check(s["n_prompt_tokens_expected"] == len(got) == enc["_n_prompt_tokens"]
                      and s["prompt_ids_sha256_expected"] == enc["_prompt_ids_sha256"] == ij.ids_sha256(got),
                      f"{job['job_id']}: sidecar token expectation differs from the processor")
                check(got.count(ids["image"]) == 256 * s["n_images"], "image token count")
                pv = enc.get("pixel_values")
                check((pv is None and s["n_images"] == 0) or tuple(pv.shape) == (s["n_images"], 3, 896, 896), "pixel_values shape")
                lengths[(config, control)] = len(got)
                n += 1
    check(lengths[("cte_p1", "none")] == 822, f"cte_p1 with one image is {lengths[('cte_p1', 'none')]} tokens, PLAN says 822")
    S["lengths"] = lengths
    return (f"{n} jobs over {len(lengths)} config x control cells: ids identical; cte_p1 none = 822 tokens; "
            f"lengths {sorted(set(lengths.values()))}")


def t_leak():
    """No label can reach a job file."""
    import re
    d, ctx = S["dirs"], S["ctx"]
    part, lab, dev = ctx["part"], ctx["lab"], set(ctx["splits"]["dev"])
    files = sorted(os.listdir(d["jobs"]))
    check(all(f.endswith(".jsonl") for f in files) and len(files) == 36, f"unexpected files in the jobs folder: {files[:5]}")
    allowed_images = dev | {e["image"] for e in ctx["examples"]} | set(bj.tile_list("smoke", ctx)[0])
    for f in files:
        text = open(os.path.join(d["jobs"], f)).read()
        for word in ("label_true", "ssa_votes", "class_a_is", "Majority", "Annotators", "image_shown", "votes"):
            check(word not in text, f"{f} contains {word!r}")
        names = {m + ".png" for m in re.findall(r"MHIST_[a-z]{3}", text)}
        check(names <= allowed_images and all(part[n] == "train" for n in names), f"{f} names a tile outside dev/examples/smoke")
        for line in text.splitlines():
            job = json.loads(line)
            check(set(job) == set(ij.JOB_KEYS), f"{f}: job keys {sorted(job)}")
            check(all(set(p) == {"type", "text"} or set(p) == {"type", "file", "sha256"} for p in job["parts"]), "part keys")
    check(not any(part[f] != "train" for f in os.listdir(os.path.join(d["bundle"], "gridded"))), "non-train tile rendered")
    check(set(os.listdir(os.path.join(d["bundle"], "gridded"))) <= allowed_images, "unexpected tile rendered")
    # 1. invariance: permute the dev labels and votes; the jobs must not change, the sidecar must
    rng = random.Random(1)
    names = sorted(dev)
    shuffled = names[:]
    rng.shuffle(shuffled)
    lab2, votes2 = dict(lab), dict(ctx["votes"])
    for a, b in zip(names, shuffled):
        lab2[a], votes2[a] = lab[b], ctx["votes"][b]
    refs = bj.ImageRefs(d["bundle"], part)
    for key in S["jobs"]:
        jobs2, side2 = bj.build(*key, "dev", ctx, refs, labels=(lab2, votes2))
        check(jobs2 == S["jobs"][key], f"{key}: job file depends on the dev labels")
        check(sum(a["label_true"] != b["label_true"] for a, b in zip(side2, S["side"][key])) > 50, "permutation did not reach the sidecar")
    # 2. no text signature tells the label: each distinct text is shared by HP and SSA tiles
    for key, jobs in S["jobs"].items():
        by_text = {}
        for job, s in zip(jobs, S["side"][key]):
            sig = tuple(p["text"] for p in job["parts"] if p["type"] == "text")
            by_text.setdefault(sig, set()).add(s["label_true"])
        check(len(by_text) == (2 if key[0].startswith("neutral") else 1), f"{key}: {len(by_text)} distinct prompt texts")
        check(all(v == {"HP", "SSA"} for v in by_text.values()), f"{key}: a prompt text is used for one label only")
    # 3. the sibling's independent audit
    import build_bundle
    check(build_bundle.audit_dir(d["jobs"]) == [], f"audit_dir flags the jobs folder: {build_bundle.audit_dir(d['jobs'])[:2]}")
    check(build_bundle.audit_dir(d["jobs"], allow=["*.jsonl"]) == [], "the jobs folder holds something that is not a job file")
    trip = build_bundle.audit_dir(d["local"])
    paired = [t for t in trip if "pair a dev tile with label information" in t]
    never = [t for t in trip if "must never be uploaded" in t]
    check(len(paired) == 30, f"audit_dir should flag exactly the 30 dev sidecars for label pairs, got {len(paired)}: {paired[:3]}")
    check(len(never) == 36 + 36 + 1, f"audit_dir should refuse every sidecar, meta file and the marker by name, got {len(never)}")
    # 4. a test-partition tile cannot be referenced
    test_tile = next(n for n in sorted(part) if part[n] == "test")
    check(exits(refs.ref, test_tile) not in (None, 0), "a test-partition tile was accepted as a job image")
    return ("36 job files: five keys only, invariant to a permutation of the dev labels, every prompt text shared by "
            f"both labels, audit_dir clean on jobs; on the sidecar folder it flags {len(paired)} files for label pairs "
            f"and {len(never)} by name; test tiles refused")


def t_validate():
    """infer_jobs rejects malformed jobs (extra keys such as a label, bad tier, absolute or parent paths)."""
    good = S["jobs"][("cte_p1", "none", "1")][0]
    bad = [{**good, "label_true": "HP"}, {**good, "tier": 1}, {**good, "seed": "7"}, {**good, "max_new_tokens": 0},
           {**good, "parts": []}, {**good, "parts": [{"type": "image", "file": "/etc/passwd"}]},
           {**good, "parts": [{"type": "image", "file": "../x.png"}]}, {**good, "parts": [{"type": "text", "text": 3}]},
           {k: v for k, v in good.items() if k != "seed"}]
    for b in bad:
        try:
            ij.validate_job(b)
        except ValueError:
            continue
        raise AssertionError(f"accepted a malformed job: {str(b)[:120]}")
    ij.validate_job(good)
    return f"{len(bad)} malformed jobs rejected"


# ------------------------------------------------------------------------------------------ import tests

def fake_output(key, tag="base", **override):
    """A Kaggle output file as infer_jobs.py would write it, with the same fabricated answers comp_run got."""
    config, control, tier = key
    d = S["dirs"]
    name = bj.job_name("dev", config, control, tier)
    sha = hashlib.sha256(open(os.path.join(d["jobs"], name + ".jsonl"), "rb").read()).hexdigest()
    rows = []
    for job, s in zip(S["jobs"][key], S["side"][key]):
        text, finish = fabricate(s["image"], config, S["ctx"]["ab"][s["image"]], s["label_true"])
        rows.append({"job_id": job["job_id"], "raw_response": text, "n_prompt_tokens": s["n_prompt_tokens_expected"],
                     "n_new_tokens": 40, "finish_reason": finish, "seconds": 1.5, "tier": tier, "seed": job["seed"],
                     "max_new_tokens": 1500, "n_images": s["n_images"], "prompt_ids_sha256": s["prompt_ids_sha256_expected"],
                     "model_revision": ij.PINNED_REVISION, "weights_verified": True, "images_verified": True,
                     "adapter_sha256": FAKE_ADAPTER_SHA,
                     "dtype": "bfloat16", "quantization": None, "batch_size": 1, "batch_rows": 1, "batch_seconds": 1.5,
                     "sampler": "per-job" if tier == "1" else "greedy", "run_signature": "sig0", "jobs_sha256": sha,
                     "run": dict(GOOD_RUN), "written_utc": "2026-10-04T00:00:00Z",
                     **override})
    path = os.path.join(TMP, "downloads", f"{name}__{tag}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_jsonl(path, rows)
    return path, rows


GOOD_TRAINED = {"protocol_run": True, "final": True, "epoch": 3, "chosen_epoch": 3, "epochs_completed": 4, "epochs_planned": 4,
                "adapter_model_sha256": FAKE_ADAPTER_SHA}
GOOD_RUN = {"gpus": ["Tesla T4"], "versions": {"transformers": "x"}, "allow_non_final_adapter": False,
            "adapter": {"sha256": FAKE_ADAPTER_SHA, "trained": GOOD_TRAINED}}
SAME_FIELDS = ("config", "control", "image", "image_shown", "label_true", "ssa_votes", "class_a_is", "fewshot",
               "prompt_sha256", "temperature", "sampling", "sampling_tag", "seed", "n_images", "thinking_trace",
               "n_valid_cited_cells", "thinking_suppressed_token_id", "raw_response", "label", "parsed_ok",
               "n_evidence", "finish_reason", "error")


def t_import():
    """Fabricated outputs round-trip into records that match comp_run's and that comp_analyze scores."""
    d = S["dirs"]
    ca.boot_bal.__defaults__ = (ca.boot_bal.__defaults__[0], 200)      # 200 bootstrap replicates here instead of 5000
    base = ["--model-tag", "fake-lora", "--local-dir", d["local"], "--out-dir", d["records"], "--train-summary", S["summary"]]
    paths = [fake_output(key)[0] for key in S["jobs"]]
    check(exits(ir.main, ["--results", *paths, "--model-tag", "fake-lora", "--local-dir", d["local"], "--out-dir", d["records"]]) == 2
          and tree_state(d["records"]) == [], "an import that names neither --train-summary nor --base-model was accepted")
    check(exits(ir.main, ["--results", *paths, *base, "--dry-run"]) is None, "dry run failed")
    check(tree_state(d["records"]) == [], "dry run wrote files")
    with quiet():
        ir.main(["--results", *paths, *base])
    n = 0
    for key, by_img in S["ref"].items():
        config, control, tier = key
        got = []
        for tiles, size in (("screen", 100), ("dev_rest", 200)):
            path = ir.record_path(d["records"], config, "fake-lora", tier, tiles, control)
            check(os.path.basename(path) == f"{config}__fake-lora__kaggle-{'tier1-t1' if tier == '1' else 'tier2-t0'}__{tiles}__{control}.jsonl",
                  f"file name {os.path.basename(path)}")
            recs = read_jsonl(path)
            check(len(recs) == size and all(r["tiles"] == tiles for r in recs), f"{path}: {len(recs)} records")
            got += recs
        for r in got:
            _, ref = by_img[r["image"]]
            check(set(ref) <= set(r), f"imported record lacks comp_run fields {sorted(set(ref) - set(r))}")
            for f in SAME_FIELDS:
                check(r[f] == ref[f], f"{key} {r['image']}: {f} = {r[f]!r}, comp_run has {ref[f]!r}")
            check(r["provider_pinned"] == r["provider_served"] == "kaggle" and r["model"] == "fake-lora" and r["cost_usd"] == 0,
                  "provider/model fields")
            check(r["usage"]["prompt_tokens"] >= 256 * r["n_images"], "usage")
            n += 1
        a = ca.summarize(sorted(got, key=lambda r: r["image"]))
        b = ca.summarize(sorted((rec for _, rec in by_img.values()), key=lambda r: r["image"]))
        check(a.pop("providers") == ["kaggle"] and b.pop("providers") == ["local"], "providers")
        check(a == b, f"{key}: comp_analyze.summarize differs between imported and comp_run records")
        check(a["n"] == 300 and 0.5 < a["parse_rate"] < 1 and a["finish_length"] > 0 and a["thinking_trace_rate"] > 0
              and (config == "co_p1" or a["cited_ge1"] > 0.5), f"{key}: fabricated answers do not exercise the scorer: {a}")
        # the step-4 format gate: valid label AND >= 1 valid grid cell, over all 300; written for the main run only
        if control == "none":
            gate = json.load(open(ir.gate_path(d["records"], config, "fake-lora", tier)))
            want = sum(r["label"] in ("HP", "SSA") and r["n_valid_cited_cells"] >= 1 for r in got) / 300
            check(gate["n"] == 300 and abs(gate["format_rate"] - want) < 1e-4 and gate["format_ok"] == (want >= 0.9)
                  and gate["adapter_sha256"] == FAKE_ADAPTER_SHA and gate["format_ok"] is False,
                  f"{key}: format gate {gate} (expected rate {want:.4f}; the fabricated answers are below the 90% bar)")
            check(want <= a["cited_ge1"] and want <= a["parse_rate"], "the format rate cannot exceed either of its two parts")
    gates = sorted(f for f, _ in tree_state(d["records"]) if f.startswith("step4_gate__"))
    check(len(gates) == 10 and all(json.load(open(os.path.join(d["records"], g)))["control"] == "none" for g in gates),
          f"one format gate file per config x tier, written by the run without an image control only: {gates}")
    ledger = ir.read_ledger(d["records"])
    check(len(ledger) == 30 and {e["adapter_sha256"] for e in ledger} == {FAKE_ADAPTER_SHA}
          and all(e["train_summary"]["status"] == "complete" and e["overrides"] == [] for e in ledger),
          f"ledger: {len(ledger)} entries")
    # comp_analyze.main() on the imported files, then on comp_run's own files: same rows apart from the provider
    same_order = {c: [r["image"] for r in read_jsonl(ir.record_path(d["records"], c, "fake-lora", "1", "screen", "none"))]
                  == [r["image"] for r in read_jsonl(os.path.join(cr.OUT, f"{c}__fake-model__local-tier1-t1__screen__none.jsonl"))]
                  for c in CONFIGS}
    rows, n_exact = {}, 0
    for label, folder in (("kaggle", d["records"]), ("local", cr.OUT)):
        ca.D = folder
        with quiet():
            ca.main()
        rows[label] = {(r["config"], r["provider"].split(" ")[1], r["control"]): r for r in json.load(open(os.path.join(folder, "RESULTS.json")))}
        check(all(r["provider"].startswith(label + " [tier") for r in json.load(open(os.path.join(folder, "RESULTS.json")))), "provider column")
    check(len(rows["kaggle"]) == len(rows["local"]) == 30, f"{len(rows['kaggle'])} analyzer rows")
    for k, r in rows["kaggle"].items():
        ref = rows["local"][k]
        check("dev" in r and r["dev"]["n"] == 300, f"{k}: analyzer did not assemble a 300-tile dev row")
        check(("dev_pass" in r) == (k[2] == "none") and ("control_ok_falls_to_chance" in r) == (k[2] != "none"), f"{k}: decision keys")
        for f in ("accuracy", "balanced_accuracy", "recall_HP", "recall_SSA", "parse_rate", "cited_ge1", "finish_length"):
            check(r["dev"][f] == ref["dev"][f], f"{k}: dev {f} {r['dev'][f]} vs {ref['dev'][f]}")
        if k[2] == "none":
            # the bootstrap CI resamples by position, so it is compared only where comp_run ran in the same order
            skip = () if same_order[k[0]] else ("balanced_boot95",)
            n_exact += not skip
            want = {**ref["screen"], "providers": ["kaggle"]}
            check("screen" in r and all(r["screen"][f] == want[f] for f in want if f not in skip), f"{k}: screen row differs")
            check(r["dev_pass"] == ref["dev_pass"] and r["screen_advances"] == ref["screen_advances"], f"{k}: gate differs")
    S["imported"] = n
    return (f"{n} records in 60 files equal comp_run's on {len(SAME_FIELDS)} fields; summarize identical; analyzer builds 30 "
            f"dev rows of 300 with the same gates; screen rows identical incl. bootstrap CI for {n_exact}/10 "
            f"(record order equals comp_run's for {sum(same_order.values())}/5 configs)")


def t_import_refusals():
    """import_results.py writes nothing when an output file is incomplete, duplicated, inconsistent or from the wrong adapter."""
    d = S["dirs"]
    key = ("cte_p1", "none", "1")
    out = os.path.join(TMP, "records_refusal")
    summary = ["--train-summary", S["summary"]]
    base = ["--model-tag", "other-tag", "--local-dir", d["local"], "--out-dir", out, *summary]
    path, rows = fake_output(key, tag="refusal")

    def attempt(mutate, extra=(), tag="other-tag", who=None):
        new = mutate([dict(r) for r in rows])
        p = os.path.join(TMP, "downloads", "mutated.jsonl")
        write_jsonl(p, new) if isinstance(new, list) else open(p, "w").write(new)
        args = ["--results", p, "--model-tag", tag, "--local-dir", d["local"], "--out-dir", out, *(summary if who is None else who), *extra]
        code = exits(ir.main, args)
        return code, tree_state(out)

    def set_field(i, k, v):
        def f(rs):
            rs[i][k] = v
            return rs
        return f

    def trained(**change):
        return {**GOOD_RUN, "adapter": {"sha256": FAKE_ADAPTER_SHA, "trained": {**GOOD_TRAINED, **change}}}

    def every(k, v):
        return lambda rs: [{**r, k: v} for r in rs]

    cases = {
        "a missing job": lambda rs: rs[:-1],
        "a duplicated job": lambda rs: rs + [rs[5]],
        "an unknown job id": lambda rs: rs + [{**rs[0], "job_id": "dev__cte_p1__none__tier1__MHIST_zzz"}],
        "a wrong seed": set_field(3, "seed", 1),
        "a wrong tier": set_field(3, "tier", "2"),
        "too few prompt tokens": set_field(7, "n_prompt_tokens", 200),
        "another token hash": set_field(7, "prompt_ids_sha256", "0" * 64),
        "another job file version": set_field(0, "jobs_sha256", "0" * 64),
        "mixed run settings": set_field(9, "run_signature", "sig1"),
        "unverified weights": set_field(2, "weights_verified", None),
        "unverified images": set_field(2, "images_verified", False),
        "no images_verified field (an older infer_jobs.py)": lambda rs: [{k: v for k, v in r.items() if k != "images_verified"} for r in rs],
        "a bad finish_reason": set_field(2, "finish_reason", "error"),
        "an adapter from a non-protocol training run": set_field(4, "run", trained(protocol_run=False)),
        "an adapter that is not final (epoch checkpoint / unfinished run)": every("run", trained(final=False)),
        "an adapter without the final flag (older train_lora.py)": every("run", {**GOOD_RUN, "adapter": {"trained": {"protocol_run": True}}}),
        "an adapter folder without step4_meta.json": every("run", {**GOOD_RUN, "adapter": {"sha256": FAKE_ADAPTER_SHA, "trained": None}}),
        "no adapter record at all (the old fixture: hash only)": every("run", {"gpus": ["Tesla T4"]}),
        "an adapter of another epoch than the chosen one": every("run", trained(epoch=2)),
        "another adapter than the training summary's": every("adapter_sha256", "cd" * 32),
        "base-model answers under --train-summary": every("adapter_sha256", None),
        "answers generated with --allow-non-final-adapter": every("run", {**trained(), "allow_non_final_adapter": True}),
        "the thinking opener in an answer": set_field(2, "raw_response", ij.THINK_OPEN + "thought"),
        "a cut-off last line": lambda rs: "".join(json.dumps(r) + "\n" for r in rs)[:-40],
        "a job with no sidecar": lambda rs: [{**r, "job_id": r["job_id"].replace("cte_p1", "nope")} for r in rs],
    }
    for what, mutate in cases.items():
        code, state = attempt(mutate)
        check(code == 2 and state == [], f"import with {what}: exit {code}, files {state[:2]}")
    n_refusals = len(cases)

    # the training summary itself: not finished, not a protocol run, not a summary
    for what, change in (("a paused training run", {"status": "paused"}), ("an in-progress training run", {"status": "in_progress"}),
                         ("a non-protocol training run", {"protocol_run": False})):
        bad = write_summary(os.path.join(TMP, "bad_summary.json"), **change)
        code, state = attempt(lambda rs: rs, who=["--train-summary", bad])
        check(code == 2 and state == [], f"import with {what}: exit {code}")
        n_refusals += 1
    open(os.path.join(TMP, "bad_summary.json"), "w").write('{"something": "else"}')
    check(attempt(lambda rs: rs, who=["--train-summary", os.path.join(TMP, "bad_summary.json")]) == (2, []), "a non-summary file was accepted")
    check(attempt(lambda rs: rs, who=["--train-summary", os.path.join(TMP, "no_such.json")]) == (2, []), "a missing summary was accepted")
    # --base-model: adapter answers are refused, adapter-free answers pass
    check(attempt(lambda rs: rs, who=["--base-model"]) == (2, []), "--base-model accepted answers generated with an adapter")
    code, state = attempt(every("adapter_sha256", None), who=["--base-model"], extra=["--dry-run"])
    check(code is None and state == [], f"--base-model refused adapter-free answers: {code}")
    n_refusals += 3

    # the sidecar must be the file build_dev_jobs.py wrote, and its labels must be those of annotations.csv
    name = bj.job_name("dev", *key)
    side_path, meta_path = (os.path.join(d["local"], name + ext) for ext in (".sidecar.jsonl", ".meta.json"))
    side_text, meta_text = open(side_path).read(), open(meta_path).read()
    side_rows = [json.loads(l) for l in side_text.splitlines()]
    flipped = dict(side_rows[0], label_true="SSA" if side_rows[0]["label_true"] == "HP" else "HP")
    edited = bj.dumps_lines([flipped] + side_rows[1:])
    try:
        open(side_path, "w").write(edited)
        check(attempt(lambda rs: rs) == (2, []), "an edited sidecar (sha256 differs from the meta file) was accepted")
        open(meta_path, "w").write(json.dumps({**json.loads(meta_text), "sidecar_sha256": hashlib.sha256(edited.encode()).hexdigest()}))
        check(attempt(lambda rs: rs) == (2, []), "a sidecar whose label differs from annotations.csv was accepted")
    finally:
        open(side_path, "w").write(side_text)
        open(meta_path, "w").write(meta_text)
    n_refusals += 2

    code, state = attempt(set_field(9, "run_signature", "sig1"), extra=["--allow-mixed-runs", "--dry-run"])
    check(code is None and state == [], "--allow-mixed-runs did not lift the mixed-run refusal")
    code, state = attempt(set_field(2, "weights_verified", None), extra=["--allow-unverified-weights", "--dry-run"])
    check(code is None and state == [], "--allow-unverified-weights did not lift the refusal")
    code, state = attempt(set_field(2, "images_verified", False), extra=["--allow-unverified-images", "--dry-run"])
    check(code is None and state == [], "--allow-unverified-images did not lift the refusal")
    code, state = attempt(set_field(4, "run", trained(protocol_run=False)), extra=["--allow-non-protocol-adapter", "--dry-run"])
    check(code is None and state == [], "--allow-non-protocol-adapter did not lift the refusal")
    code, state = attempt(every("run", trained(final=False)), extra=["--allow-non-protocol-adapter", "--dry-run"])
    check(code == 2 and state == [], "a non-final adapter was let through by --allow-non-protocol-adapter")

    files = lambda: sorted(f for f, _ in tree_state(out))                                    # noqa: E731
    check(exits(ir.main, ["--results", path, *base]) is None, "clean import failed")
    gate_name = os.path.basename(ir.gate_path(out, "cte_p1", "other-tag", "1"))
    check(files() == sorted(["cte_p1__other-tag__kaggle-tier1-t1__screen__none.jsonl", "cte_p1__other-tag__kaggle-tier1-t1__dev_rest__none.jsonl",
                             gate_name, ir.LEDGER]), f"clean import wrote {files()}")
    before = {f: open(os.path.join(out, f)).read() for f in files()}
    check(exits(ir.main, ["--results", path, *base]) == 2, "a second import over existing files was not refused")
    check({f: open(os.path.join(out, f)).read() for f in files()} == before, "refused import changed files")
    check(exits(ir.main, ["--results", path, *base, "--force"]) is None, "--force did not replace the files")
    check(len(ir.read_ledger(out)) == 2 and ir.read_ledger(out)[1]["overrides"] == ["force"], "ledger after --force")
    check(exits(ir.main, ["--results", path, "--model-tag", "bad__tag", "--local-dir", d["local"], "--out-dir", out, *summary]) == 2, "bad tag accepted")

    # one adapter per configuration, tier and control: a second, different adapter is refused whatever the model tag
    other_sha = "cd" * 32
    other_summary = write_summary(os.path.join(TMP, "other_summary.json"),
                                  best_adapter={"adapter_model_sha256": other_sha, "final": True})
    other_run = {**GOOD_RUN, "adapter": {"sha256": other_sha, "trained": {**GOOD_TRAINED, "adapter_model_sha256": other_sha}}}
    second = lambda rs: [{**r, "adapter_sha256": other_sha, "run": other_run} for r in rs]    # noqa: E731
    n_before = len(tree_state(out))
    code, state = attempt(second, tag="second-adapter", who=["--train-summary", other_summary])
    check(code == 2 and len(state) == n_before, f"a second, different adapter for the same dev run was imported: {code}")
    code, state = attempt(second, tag="second-adapter", who=["--train-summary", other_summary], extra=["--allow-another-adapter"])
    check(code is None and len(state) == n_before + 3 and ir.read_ledger(out)[-1]["overrides"] == ["allow_another_adapter"],
          f"--allow-another-adapter did not import and record the second adapter: {code}")
    n_refusals += 1

    # the format gate fails loudly but the (spent) dev run is still recorded; a passing run says ok
    good_answer = json.dumps({"label": "HP", "confidence": 0.7, "evidence": [{"feature": "serration", "grid_cells": ["B2"], "description": "x"}]})
    out_gate = os.path.join(TMP, "records_gate")
    for tag, answer, ok in (("fmt-ok", good_answer, True), ("fmt-bad", json.dumps({"label": "HP", "confidence": 0.7, "evidence": []}), False)):
        p = os.path.join(TMP, "downloads", f"{tag}.jsonl")
        write_jsonl(p, [{**r, "raw_response": answer, "finish_reason": "stop"} for r in rows])
        check(exits(ir.main, ["--results", p, "--model-tag", tag, "--local-dir", d["local"], "--out-dir", out_gate, *summary]) is None, f"{tag} import")
        gate = json.load(open(ir.gate_path(out_gate, "cte_p1", tag, "1")))
        check(gate["format_ok"] is ok and gate["format_rate"] == (1.0 if ok else 0.0) and gate["n"] == 300, f"{tag}: gate {gate}")
    ca_d, ca.D = ca.D, out_gate                       # comp_analyze still reads the folder: the ledger and gate files are not *.jsonl
    try:
        with quiet():
            ca.main()
    finally:
        ca.D = ca_d
    check(len(json.load(open(os.path.join(out_gate, "RESULTS.json")))) == 2, "comp_analyze did not read the folder with gate and ledger files")

    # smoke answers go to smoke/, which the analyzer never reads; they need no final adapter; the label-free check decides the exit code
    name = bj.job_name("smoke", "cte_p1", "none", "1")
    jobs, sha = ij.read_jobs(os.path.join(d["jobs"], name + ".jsonl"))
    side = read_jsonl(os.path.join(d["local"], name + ".sidecar.jsonl"))
    pool = set(S["ctx"]["splits"]["fewshot_pool"])
    check(len(jobs) == 20 and all(s["image"] in pool and s["tiles"] == "smoke" for s in side), "smoke set")
    smoke = [{**rows[0], "job_id": j["job_id"], "seed": j["seed"], "jobs_sha256": sha, "raw_response": good_answer + "<end_of_turn>",
              "finish_reason": "stop", "run": trained(protocol_run=False, final=False)} for j in jobs]
    p = os.path.join(TMP, "downloads", "smoke.jsonl")
    write_jsonl(p, smoke)
    debug_summary = write_summary(os.path.join(TMP, "debug_summary.json"), protocol_run=False)
    smoke_base = ["--model-tag", "other-tag", "--local-dir", d["local"], "--out-dir", out, "--train-summary", debug_summary]
    n_ledger = len(ir.read_ledger(out))
    check(exits(ir.main, ["--results", p, *smoke_base]) is None, "smoke import with a debug adapter failed")
    check(os.path.exists(os.path.join(out, "smoke", "cte_p1__other-tag__kaggle-tier1-t1__smoke__none.jsonl")), "smoke file not in smoke/")
    check(len(ir.read_ledger(out)) == n_ledger, "a smoke import was written to the dev ledger")
    no_cell = json.dumps({"label": "SSA", "confidence": 0.5, "evidence": []})
    for what, change, which in (("one answer cut off at the token limit", {"finish_reason": "length"}, {0}),
                                ("3 of 20 answers without a grid cell (85%)", {"raw_response": no_cell}, {0, 1, 2})):
        write_jsonl(p, [dict(r, **change) if n in which else r for n, r in enumerate(smoke)])
        again = ["--results", p, "--model-tag", "smoke-again", *smoke_base[2:], "--dry-run"]       # a tag with no file yet
        check(exits(ir.main, again) == 1, f"the smoke format check passed with {what}")
    write_jsonl(p, smoke)
    check(exits(ir.main, again) is None, "the smoke format check failed on 20 good answers")
    write_jsonl(p, [{**r, "adapter_sha256": other_sha} for r in smoke])
    check(exits(ir.main, again) == 2, "smoke answers from another adapter than the summary's were accepted")

    # --control-tiles dev: an image control as ONE dev file in comp_run's order -> the analyzer row is comp_run's own
    out2 = os.path.join(TMP, "records_control_dev")
    for config in ("cte_p1", "neutral_crit_fs"):
        key = (config, "mismatch", "1")
        path2, _ = fake_output(key, tag="ctl")
        check(exits(ir.main, ["--results", path2, "--model-tag", "fake-lora", "--local-dir", d["local"], "--out-dir", out2,
                              "--control-tiles", "dev", *summary]) is None, "--control-tiles dev import failed")
        got = read_jsonl(os.path.join(out2, f"{config}__fake-lora__kaggle-tier1-t1__dev__mismatch.jsonl"))
        ref = read_jsonl(os.path.join(cr.OUT, f"{config}__fake-model__local-tier1-t1__dev__mismatch.jsonl"))
        check(len(got) == 300 and all(r["tiles"] == "dev" for r in got) and [r["image"] for r in got] == [r["image"] for r in ref],
              f"{config}: dev control file is not in comp_run's order")
        a, b = ca.summarize(got), ca.summarize(ref)
        check({**a, "providers": 0} == {**b, "providers": 0}, f"{config}: control summary differs from comp_run's incl. bootstrap CI")
    check(sorted(f for f, _ in tree_state(out2)) == sorted(["cte_p1__fake-lora__kaggle-tier1-t1__dev__mismatch.jsonl",
                                                            "neutral_crit_fs__fake-lora__kaggle-tier1-t1__dev__mismatch.jsonl", ir.LEDGER]),
          "unexpected files for --control-tiles dev (an image control writes no format gate)")
    return (f"{n_refusals} refusals fire with nothing written (incl. non-final / wrong / missing adapter, unfinished training, edited "
            "sidecar, unverified images, a second adapter); overrides, --force and the ledger work; the format gate is written "
            "(ok and FAILS) and comp_analyze still reads the folder; smoke answers accept a debug adapter and the label-free "
            "check sets the exit code; --control-tiles dev reproduces comp_run's control file order and its bootstrap CI")


# ------------------------------------------------------------------------------------------ infer tests

def t_pinned():
    """The hash constants in infer_jobs.py are the official files' hashes."""
    n = 0
    for name, want in ij.PINNED_GIT_BLOB.items():
        check(ij.git_blob_sha1(os.path.join(MODEL_DIR, name)) == want, f"{name}: git blob hash differs from the constant")
        n += 1
    check(ij.sha256_file(os.path.join(MODEL_DIR, "tokenizer.json")) == ij.PINNED_SHA256["tokenizer.json"], "tokenizer.json sha256")
    for name in ij.WEIGHT_FILES:                       # compared with the download metadata; the weights are not read
        meta = os.path.join(MODEL_DIR, ".cache", "huggingface", "download", name + ".metadata")
        check(os.path.exists(meta), f"no download metadata for {name}")
        revision, etag = open(meta).read().split()[:2]
        check(revision == ij.PINNED_REVISION and etag == ij.PINNED_SHA256[name], f"{name}: constant differs from the download metadata")
        n += 1
    with quiet():
        fp = ij.fingerprint_model(MODEL_DIR, None, hash_weights=False)
    check(fp["verified"] is None and not fp["mismatch"] and not fp["missing"], f"official folder does not fingerprint clean: {fp}")
    return f"{n + 1} pinned hashes confirmed; official folder fingerprints clean with weight hashing skipped"


def t_sampler():
    """The per-job sampler is stock HF sampling with a private generator per row."""
    import torch
    from transformers import LogitsProcessorList, NoBadWordsLogitsProcessor, TopKLogitsWarper, TopPLogitsWarper
    cls = ij.per_job_sampler_class()
    g = torch.Generator().manual_seed(5)
    vocab, banned, eos = 4000, 100, [1, 106]
    logits = torch.randn(3, vocab, generator=g) * 2.5
    logits[:, banned] = 50.0                                            # the model "wants" the banned token
    logits[2, :] = torch.round(logits[2, :] * 2) / 2                    # many exact ties, as bf16 logits have
    dummy = torch.zeros((3, 4), dtype=torch.long)
    stock = LogitsProcessorList([NoBadWordsLogitsProcessor([[banned]], eos_token_id=eos),
                                 TopKLogitsWarper(top_k=64, min_tokens_to_keep=1), TopPLogitsWarper(top_p=0.95, min_tokens_to_keep=1)])
    n_draws = 6000
    for row in range(3):
        want = torch.softmax(stock(dummy[row:row + 1], logits[row:row + 1].clone())[0].double(), dim=0)
        s = cls([11], 1.0, 64, 0.95, banned_ids=[banned], eos_ids=[])
        idx, probs = s.kept(dummy[row:row + 1], logits[row:row + 1])
        dense = torch.zeros(vocab, dtype=torch.float64)
        dense[idx] = probs
        check(set(idx.tolist()) == set(torch.nonzero(want).squeeze(1).tolist()), f"row {row}: kept tokens differ from stock HF warpers")
        check(torch.allclose(dense, want, atol=1e-9), f"row {row}: probabilities differ from stock HF")
        check(banned not in idx.tolist() and 2 <= idx.numel(), "banned token kept, or a degenerate test distribution")
        counts = torch.zeros(vocab, dtype=torch.float64)
        for _ in range(n_draws):
            out = s(dummy[row:row + 1], logits[row:row + 1])
            check(int(torch.isfinite(out).sum()) == 1, "sampler must leave exactly one finite score")
            counts[int(out.argmax())] += 1
        check(counts[banned] == 0 and set(torch.nonzero(counts).squeeze(1).tolist()) <= set(idx.tolist()), "drew outside the kept set")
        sigma = torch.sqrt(want * (1 - want) / n_draws)
        z = ((counts / n_draws - want).abs() / sigma.clamp(min=1e-12))[want > 0].max().item()
        check(z < 4.5, f"row {row}: empirical frequencies are {z:.1f} sigma from the stock distribution")
    # a row's draws depend on its own seed only: alone, first in a batch, last in a batch
    def draws(seeds, rows, steps=200):
        s = cls(seeds, 1.0, 64, 0.95, banned_ids=[banned], eos_ids=[])
        x = torch.stack([logits[r] for r in rows])
        return [[int(t) for t in s(dummy[:len(rows)], x).argmax(dim=1)] for _ in range(steps)]
    alone = [d[0] for d in draws([7], [0])]
    check(alone == [d[0] for d in draws([7, 8, 9], [0, 1, 2])], "draws change when other rows join the batch")
    check(alone == [d[2] for d in draws([9, 8, 7], [2, 1, 0])], "draws change with the row position")
    check(alone == [d[0] for d in draws([7], [0])] and alone != [d[0] for d in draws([8], [0])], "seeding")
    check(len(set(alone)) > 5, "degenerate draws")
    # finished rows stop consuming their generator
    s = cls([1, 2], 1.0, 64, 0.95, banned_ids=[banned], eos_ids=[5])
    forced = torch.full((2, vocab), -float("inf"))
    forced[0, 5], forced[1, 9] = 0.0, 0.0
    s(dummy[:2], forced)
    s(dummy[:2], forced)
    check(s.finished == [True, False] and s.n_draws == [1, 2], f"finished-row bookkeeping {s.finished} {s.n_draws}")
    return (f"kept set and probabilities equal transformers' TopK/TopP on 3 rows (one with ties); {n_draws} draws per row "
            "within 4.5 sigma; banned token never drawn; a row's draws are identical alone and inside any batch")


def make_tiny_model(path):
    import torch
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration
    cfg = Gemma3Config(
        text_config=dict(vocab_size=262208, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=16, sliding_window=1024, query_pre_attn_scalar=16,
                         max_position_embeddings=131072),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=1,
                           image_size=896, patch_size=14, num_channels=3, vision_use_head=False),
        mm_tokens_per_image=256, boi_token_index=255999, eoi_token_index=256000, image_token_index=262144,
        eos_token_id=[1, 106])
    torch.manual_seed(0)
    with quiet():
        Gemma3ForConditionalGeneration(cfg).save_pretrained(path)
    for f in os.listdir(MODEL_DIR):                    # the REAL tokenizer / processor / chat template
        if f.startswith(("tokenizer", "special_tokens", "added_tokens", "preprocessor", "processor", "chat_template")):
            os.symlink(os.path.join(MODEL_DIR, f), os.path.join(path, f))


def t_tiny():
    """infer_jobs.py end to end on CPU: a tiny random Gemma3 model behind the real processor."""
    import torch
    d = S["dirs"]
    tiny = os.path.join(TMP, "tiny")
    make_tiny_model(tiny)
    work = os.path.join(TMP, "working")
    os.makedirs(work)
    # a small mixed job file: 1-image, text-only and few-shot (7 images) prompts, both tiers, 10 new tokens each
    pick = lambda key, n: [dict(j, max_new_tokens=10) for j in S["jobs"][key][:n]]                    # noqa: E731
    jobs = pick(("cte_p1", "none", "1"), 3) + pick(("cte_p1", "noimage", "1"), 2) + pick(("cte_p1", "mismatch", "1"), 1) \
        + pick(("neutral_crit_fs", "none", "1"), 1) + pick(("cte_p1", "none", "2"), 2) + pick(("neutral_cte", "noimage", "2"), 1)
    jobs_path = os.path.join(work, "mini.jsonl")
    write_jsonl(jobs_path, jobs)
    side = {s["job_id"]: s for key in S["side"] for s in S["side"][key]}
    common = ["--model-dir", tiny, "--bundle-dir", d["bundle"], "--device", "cpu"]

    def run(out, *extra, jobs_file=jobs_path, expect=0):
        argv = ["--jobs", jobs_file, "--out", os.path.join(work, out), *common, "--allow-unverified-weights", *extra]
        code = main_code(ij.main, argv)
        check(code == expect, f"infer_jobs {out} {extra}: exit code {code}, expected {expect}")
        return {r["job_id"]: r for r in read_jsonl(os.path.join(work, out))} if os.path.exists(os.path.join(work, out)) else {}

    notes = []
    # weights that are not the pinned revision are refused before anything is loaded
    code = main_code(ij.main, ["--jobs", jobs_path, "--out", os.path.join(work, "refused.jsonl"), *common])
    check(code == 2 and not os.path.exists(os.path.join(work, "refused.jsonl")), "unverified weights were not refused")
    check(main_code(ij.main, ["--jobs", jobs_path, *common, "--dry-run"]) == 0, "dry run failed")

    a = run("a.jsonl")
    check(list(a) == [j["job_id"] for j in jobs], "output order or count")
    need = {"job_id", "raw_response", "n_prompt_tokens", "n_new_tokens", "finish_reason", "seconds", "tier", "seed",
            "model_revision", "adapter_sha256", "dtype", "weights_verified", "images_verified", "batch_size", "sampler",
            "run_signature", "jobs_sha256", "prompt_ids_sha256", "n_images", "max_new_tokens", "run"}
    for job in jobs:
        r, s = a[job["job_id"]], side[job["job_id"]]
        check(need <= set(r), f"output line lacks {sorted(need - set(r))}")
        check(r["n_prompt_tokens"] == s["n_prompt_tokens_expected"] and r["prompt_ids_sha256"] == s["prompt_ids_sha256_expected"],
              f"{job['job_id']}: prompt tokens differ from the sidecar expectation")
        check(r["finish_reason"] in ("stop", "length") and (r["finish_reason"] == "stop" or r["n_new_tokens"] == 10), "finish/length")
        check(r["tier"] == job["tier"] and r["seed"] == job["seed"] and r["n_images"] == s["n_images"], "tier/seed/n_images")
        check(r["dtype"] == "float32" and r["weights_verified"] is False and r["adapter_sha256"] is None
              and r["model_revision"] == ij.PINNED_REVISION and r["sampler"] == ("per-job" if job["tier"] == "1" else "greedy"),
              "provenance fields")
        check(r["images_verified"] is True and r["run"]["images_verified"] is True and r["run"]["allow_non_final_adapter"] is False
              and r["run"]["adapter_is_final_protocol_adapter"] is None and r["run"]["token_type_ids_source"].startswith("processor")
              and r["run"]["precision"]["quantized_scope"] is None, f"run record {r['run']}")
        check(isinstance(r["raw_response"], str) and r["raw_response"] and "<pad>" not in r["raw_response"], "raw_response")
    check(len({a[j["job_id"]]["raw_response"] for j in jobs if j["tier"] == "1"}) >= 5, "tier-1 answers are not distinct")
    notes.append(f"{len(jobs)} jobs (0, 1 and 7 images; tiers 1 and 2) generated, fields complete, prompt token hashes equal the sidecar's")

    # resume: three finished lines plus a line cut off by a crash
    lines = open(os.path.join(work, "a.jsonl")).read().splitlines(keepends=True)
    open(os.path.join(work, "b.jsonl"), "w").write("".join(lines[:3]) + lines[3][:150])
    b = run("b.jsonl")
    check({k: v["raw_response"] for k, v in b.items()} == {k: v["raw_response"] for k, v in a.items()}, "resumed run differs")
    check(open(os.path.join(work, "b.jsonl")).read().splitlines()[:3] == [l.rstrip("\n") for l in lines[:3]], "finished lines were rewritten")
    before = open(os.path.join(work, "b.jsonl")).read()
    run("b.jsonl")
    check(open(os.path.join(work, "b.jsonl")).read() == before, "a complete file was touched on re-run")
    run("b.jsonl", "--batch-size", "2", expect=2)                      # other settings: refuses to continue the file
    run("b.jsonl", "--no-verify-images", expect=2)                     # image verification is part of the run signature
    nv = run("nv.jsonl", "--no-verify-images")
    check(all(r["images_verified"] is False for r in nv.values()) and {k: v["raw_response"] for k, v in nv.items()}
          == {k: v["raw_response"] for k, v in a.items()} and next(iter(nv.values()))["run_signature"] != a[jobs[0]["job_id"]]["run_signature"],
          "--no-verify-images is not recorded in the output and the run signature")
    notes.append("resume redoes only the cut-off job and reproduces the rest; a finished file is left alone; changed settings "
                 "(batch size, image verification) are refused; --no-verify-images is recorded as images_verified=false")
    status = json.load(open(os.path.join(work, "a.status.json")))
    check(status["status"] == "complete" and status["n_done"] == status["n_jobs"] == len(jobs), f"status file {status}")

    # a new session: /kaggle/working is empty, the earlier output is attached read-only and passed with --resume-from
    prev = os.path.join(work, "previous_session", "nested")
    os.makedirs(prev)
    open(os.path.join(prev, "r.jsonl"), "w").write("".join(lines[:4]))
    open(os.path.join(prev, "r.jsonl.shard1of2"), "w").write(lines[6])
    open(os.path.join(prev, "other.jsonl"), "w").write(lines[8])               # another file name: not picked up
    budget = ["--resume-from", os.path.dirname(prev), "--time-budget-hours", "1e-9"]
    stopped = run(os.path.join("s2", "r.jsonl"), *budget, "--exit-zero", "no", expect=ij.EXIT_TIME_BUDGET)
    check(len(stopped) == 5, f"{len(stopped)} answers carried over, expected 5")
    status = json.load(open(os.path.join(work, "s2", "r.status.json")))
    check(status["status"] == "incomplete" and status["n_done"] == 5 and status["exit_code"] == ij.EXIT_TIME_BUDGET, f"status {status}")
    run(os.path.join("s2", "r.jsonl"), *budget, "--exit-zero", "yes", expect=0)   # on Kaggle: keep the partial output
    check(json.load(open(os.path.join(work, "s2", "r.status.json")))["status"] == "incomplete", "exit-zero hid the status")
    r = run(os.path.join("s2", "r.jsonl"), "--resume-from", os.path.dirname(prev))
    check({k: v["raw_response"] for k, v in r.items()} == {k: v["raw_response"] for k, v in a.items()}, "resumed session differs")
    check(all(r[json.loads(l)["job_id"]] == json.loads(l) for l in lines[:4] + [lines[6]]), "carried answers were regenerated")
    check(main_code(ij.main, ["--jobs", jobs_path, "--out", os.path.join(work, "s3", "r.jsonl"), *common, "--allow-unverified-weights",
                              "--time-budget-hours", "1e-9", "--exit-zero", "yes"]) == ij.EXIT_TIME_BUDGET,
          "an early stop with nothing written must not exit 0")
    notes.append("--resume-from carries 5 finished answers from an earlier session's folder and regenerates none of them; a time-budget "
                 "stop exits 3, or 0 with --exit-zero yes only when answers exist; the status file says incomplete")

    # job order does not matter
    rev_path = os.path.join(work, "mini_reversed.jsonl")
    write_jsonl(rev_path, jobs[::-1])
    c = run("c.jsonl", jobs_file=rev_path)
    check({k: v["raw_response"] for k, v in c.items()} == {k: v["raw_response"] for k, v in a.items()}, "job order changes answers")
    # two workers: shard files, then merged in job order
    w = run("w.jsonl", "--workers", "2")
    check(list(w) == [j["job_id"] for j in jobs] and not ij.shard_files(os.path.join(work, "w.jsonl")), "worker merge")
    check({k: v["raw_response"] for k, v in w.items()} == {k: v["raw_response"] for k, v in a.items()}, "workers change answers")
    notes.append("reversed job order and 2 workers give the same answer for every job")

    # stock HF sampling path: allowed at batch size 1, deterministic, refused with a batch
    h1, h2 = run("h1.jsonl", "--sampler", "hf"), run("h2.jsonl", "--sampler", "hf")
    check({k: v["raw_response"] for k, v in h1.items()} == {k: v["raw_response"] for k, v in h2.items()}, "--sampler hf is not deterministic")
    check(all(h1[j["job_id"]]["raw_response"] == a[j["job_id"]]["raw_response"] for j in jobs if j["tier"] == "2"), "greedy differs")
    check(main_code(ij.main, ["--jobs", jobs_path, *common, "--sampler", "hf", "--batch-size", "2"]) == 2, "--sampler hf with a batch accepted")

    # batch size 3 vs 1: reported, not asserted (float-level batch dependence is documented)
    b3 = run("b3.jsonl", "--batch-size", "3")
    same = sum(b3[k]["raw_response"] == a[k]["raw_response"] for k in a)
    check(len(b3) == len(a) and all(b3[k]["n_prompt_tokens"] == a[k]["n_prompt_tokens"] for k in a), "batched run incomplete")
    with quiet():
        cmp_code = ij.compare(os.path.join(work, "a.jsonl"), os.path.join(work, "b3.jsonl"))
    check((cmp_code == 0) == (same == len(a)), "--compare disagrees")
    notes.append(f"batch size 3 vs 1 on this CPU, fp32, random tiny model: {same}/{len(a)} answers identical (reported, not required)")

    # in-process: forced end of turn, and the thinking-token ban
    from transformers import LogitsProcessor
    proc = S.get("processor") or ij.load_processor(tiny)
    ids = ij.special_ids(proc)
    args = types.SimpleNamespace(model_dir=tiny, device="cpu", adapter_dir=None, attn_implementation="eager",
                                 precision="auto", compute_dtype="auto")
    precision = ij.resolve_precision(args, None, {"cuda": False})
    check(precision["mode"] == "fp32" and precision["quantization"] is None, f"CPU precision {precision}")
    with quiet():
        model, device, dtype = ij.load_model(args, ij.check_env(), precision)

    class ForceEos(LogitsProcessor):
        def __init__(self, steps):
            self.steps, self.start, self.first = steps, None, None

        def __call__(self, input_ids, scores):
            if self.start is None:
                self.start, self.first = input_ids.shape[1], scores.argmax(dim=-1).tolist()
            step = input_ids.shape[1] - self.start
            for row, at in enumerate(self.steps):
                if at is not None and step == at:
                    scores = scores.clone()
                    scores[row, :] = -float("inf")
                    scores[row, 106] = 0.0
            return scores

    def gen(batch_jobs, hook=None):
        ij._TEST_LOGITS_PROCESSORS[:] = [hook] if hook else []
        try:
            encs = [ij.encode_job(proc, j, d["bundle"], ids) for j in batch_jobs]
            return ij.generate_batch(model, proc, batch_jobs, encs, device, dtype, "per-job", ids)
        finally:
            ij._TEST_LOGITS_PROCESSORS[:] = []

    for tier_jobs in ([jobs[0], jobs[3]], [jobs[7], jobs[8]]):                 # tier 1 (1 image + text only), tier 2
        res = gen(tier_jobs, ForceEos([2, None]))
        check(res[0]["finish_reason"] == "stop" and res[0]["n_new_tokens"] == 3 and res[0]["raw_response"].endswith("<end_of_turn>"),
              f"forced end of turn: {res[0]}")
        check(res[1]["finish_reason"] == "length" and res[1]["n_new_tokens"] == 10 and "<pad>" not in res[0]["raw_response"]
              and "<pad>" not in res[1]["raw_response"], f"row after a finished row: {res[1]}")
    notes.append("an answer that ends early is cut at <end_of_turn> (stop, no padding) while its batch neighbour runs to the limit (length)")

    for job in (jobs[7], jobs[0]):                                             # greedy, then sampled
        probe = ForceEos([None])
        base = gen([job], probe)[0]
        first = probe.first[0]                                                 # the token the model picks first
        check(first != ij.THINK_OPEN_ID, "token 100 reached the custom processors: the ban is not applied first")
        old = ij.THINK_OPEN_ID
        ij.THINK_OPEN_ID = first                                               # pretend that token opens a thinking trace
        try:
            probe2 = ForceEos([None])
            banned = gen([job], probe2)[0]
        finally:
            ij.THINK_OPEN_ID = old
        check(probe2.first[0] != first and banned["raw_response"] != base["raw_response"],
              "bad_words_ids did not remove the banned token from generation")
    notes.append("bad_words_ids removes a banned token even when it is the model's first choice (greedy and sampled)")
    del model

    # pieces that only matter on Kaggle: precision choice, adapter fingerprint, kernel start-up
    t4 = {"cuda": True, "gpus": ["Tesla T4"], "native_bf16": False}
    l4 = {"cuda": True, "gpus": ["NVIDIA L4"], "native_bf16": True}
    ns = lambda **k: types.SimpleNamespace(**{"device": "cuda", "precision": "auto", "compute_dtype": "auto", **k})   # noqa: E731
    trained_4bit = {"trained": {"precision": {"mode": "4bit", "compute_dtype": "float32", "used_4bit": True,
                                              "quantized_scope": ij.QUANTIZED_SCOPE}}}
    trained_bf16 = {"trained": {"precision": {"mode": "bf16", "compute_dtype": "bfloat16", "used_4bit": False}}}
    p = ij.resolve_precision(ns(), trained_4bit, l4)
    check((p["mode"], p["compute_dtype"], p["quantization"], p["as_trained"]) == ("4bit", "float32", "bnb-4bit-nf4-double-quant", True), f"{p}")
    check(p["quantized_scope"] == ij.QUANTIZED_SCOPE and p["vision_tower"] == "float32, not quantised", f"4-bit scope record {p}")
    old_scope = {"trained": {"precision": {"mode": "4bit", "compute_dtype": "float32", "used_4bit": True}}}   # vision tower was quantised too
    check(exits(ij.resolve_precision, ns(), old_scope, t4) == 2, "an adapter trained with another 4-bit scope was accepted")
    p = ij.resolve_precision(ns(), trained_bf16, t4)
    check((p["mode"], p["compute_dtype"], p["quantization"], p["as_trained"]) == ("bf16", "bfloat16", None, True), f"{p}")
    p = ij.resolve_precision(ns(), None, t4)
    check((p["mode"], p["compute_dtype"], p["as_trained"]) == ("4bit", "float32", None), f"{p}")
    p = ij.resolve_precision(ns(), None, l4)
    check((p["mode"], p["compute_dtype"]) == ("bf16", "bfloat16"), f"{p}")
    p = ij.resolve_precision(ns(precision="bf16"), trained_4bit, t4)
    check(p["mode"] == "bf16" and p["as_trained"] is False and "EMULATED" in p["reason"], f"{p}")
    p = ij.resolve_precision(ns(precision="fp32"), None, t4)
    check((p["mode"], p["weights_dtype"], p["compute_dtype"]) == ("fp32", "float32", "float32"), f"{p}")

    adapter_dir = os.path.join(work, "dataset", "best_adapter")
    os.makedirs(adapter_dir)
    open(os.path.join(adapter_dir, "adapter_model.safetensors"), "wb").write(b"not a real adapter")
    adapter_sha = hashlib.sha256(b"not a real adapter").hexdigest()
    json.dump({"r": 16, "lora_alpha": 16, "peft_type": "LORA"}, open(os.path.join(adapter_dir, "adapter_config.json"), "w"))
    meta_path = os.path.join(adapter_dir, "step4_meta.json")
    meta = {"epoch": 3, "precision": trained_4bit["trained"]["precision"], "image_processor_class": "X", "protocol_run": True,
            "final": False, "epochs_planned": 4, "base_model": {"id": ij.PINNED_MODEL, "revision": ij.PINNED_REVISION}}
    json.dump(meta, open(meta_path, "w"))
    found = ij.find_dir_with(os.path.join(work, "dataset"), "adapter_config.json", "adapter")
    fa = ij.fingerprint_adapter(found)
    check(found == adapter_dir and fa["sha256"] == adapter_sha and fa["r"] == 16
          and fa["trained"]["precision"]["mode"] == "4bit" and fa["trained"]["epoch"] == 3 and fa["trained"]["final"] is False,
          f"adapter fingerprint {fa}")

    # the adapter gate: dev jobs run only with the final adapter of a protocol run
    dev_jobs, smoke_jobs = jobs[:2], [dict(jobs[0], job_id="smoke__cte_p1__none__tier1__MHIST_xxx")]
    final_meta = {**meta, "final": True, "chosen_epoch": 3, "epochs_completed": 4, "adapter_model_sha256": adapter_sha}

    def gate(m, js, allow=False):
        if m is None:
            os.remove(meta_path) if os.path.exists(meta_path) else None
        else:
            json.dump(m, open(meta_path, "w"))
        return exits(lambda: S.__setitem__("gate", ij.adapter_gate(ij.fingerprint_adapter(adapter_dir), js, allow)))

    check(ij.adapter_gate(None, dev_jobs) == [], "no adapter: nothing to gate")
    check(gate(final_meta, dev_jobs) is None and S["gate"] == [], "the final adapter of a protocol run was refused for dev jobs")
    for what, m in (("best-so-far adapter of an unfinished run / an epoch checkpoint (final=false)", meta),
                    ("a debug run's adapter (protocol_run=false)", {**final_meta, "protocol_run": False}),
                    ("an older adapter without the final flag", {k: v for k, v in meta.items() if k != "final"}),
                    ("a folder without step4_meta.json", None),
                    ("final without a recorded sha256", {**final_meta, "adapter_model_sha256": None})):
        check(gate(m, dev_jobs) == 2, f"dev jobs were accepted with {what}")
        check(gate(m, smoke_jobs) is None and S["gate"], f"smoke jobs (pool tiles) were refused with {what}")
        check(gate(m, dev_jobs, allow=True) is None and S["gate"], f"--allow-non-final-adapter did not let {what} through")
    check(gate({**final_meta, "adapter_model_sha256": "0" * 64}, smoke_jobs) == 2, "a meta file from another adapter (sha256 differs) was accepted")
    # through the command line: refused before anything is read; the final adapter passes the dry run
    json.dump(meta, open(meta_path, "w"))
    with_adapter = ["--jobs", jobs_path, *common, "--adapter-dir", os.path.join(work, "dataset"), "--dry-run"]
    check(main_code(ij.main, with_adapter) == 2, "infer_jobs ran dev jobs with a non-final adapter")
    check(main_code(ij.main, [*with_adapter, "--allow-non-final-adapter"]) == 0, "--allow-non-final-adapter did not override")
    json.dump(final_meta, open(meta_path, "w"))
    check(main_code(ij.main, with_adapter) == 0, "infer_jobs refused the final protocol adapter")
    a_args = types.SimpleNamespace(allow_non_final_adapter=True, adapter_dir=adapter_dir, skip_weight_hash=False, allow_unverified_weights=False,
                                   allow_mixed_runs=False, no_verify_images=False, limit=None, resume_from=None, jobs="j", model_dir="m",
                                   bundle_dir="b", model_revision="r", precision="auto", compute_dtype="auto", attn_implementation="eager",
                                   image_backend="default", batch_size=1, sampler="per-job", device="cuda", time_budget_hours=11.0)
    check("--allow-non-final-adapter" in ij.child_argv(a_args, "s.py", "o.jsonl", 0, 2), "worker processes do not inherit --allow-non-final-adapter")

    # 4-bit scope: the skip list against this transformers version's own matching rule, and the check on a loaded model
    with quiet():
        scope_model, _, _ = ij.load_model(args, ij.check_env(), precision)
    linear = [n for n, m in scope_model.named_modules() if type(m) is torch.nn.Linear]
    lm = [n for n in linear if "language_model" in n.split(".")]
    check(len(lm) == 14 and len(linear) > len(lm) + 1, f"tiny model linear layers: {len(lm)} / {len(linear)}")
    try:
        from transformers.quantizers.quantizers_utils import should_convert_module
        converted = [n for n in linear if should_convert_module(n, list(ij.BNB_SKIP_MODULES))]
        check(sorted(converted) == sorted(lm), f"this transformers would quantise {sorted(set(converted) - set(lm))[:3]} and skip "
                                               f"{sorted(set(lm) - set(converted))[:3]} with BNB_SKIP_MODULES")
        alone = [n for n in linear if should_convert_module(n, ["vision_tower", "multi_modal_projector", "lm_head"])]
        rule = f"installed rule: skip list exact; 'vision_tower' alone would quantise {len(set(alone) - set(lm))} vision layers"
    except ImportError:
        rule = "installed transformers has no should_convert_module (older matching rule)"
    old_rule = [n for n in linear if not any((k + "." in n) or k == n for k in ij.BNB_SKIP_MODULES)]      # transformers 4.x: substring
    check(sorted(old_rule) == sorted(lm), "BNB_SKIP_MODULES does not isolate the language model under the 4.x substring rule")

    class Linear4bit(torch.nn.Linear):                                         # stands in for bitsandbytes' class (matched by name)
        pass

    mods = dict(scope_model.named_modules())
    for n in lm:
        mods[n].__class__ = Linear4bit
    check(ij.check_quantized_scope(scope_model) == 14, "check_quantized_scope on a correctly quantised model")
    vision = next(n for n in linear if "vision_tower" in n.split("."))
    mods[vision].__class__ = Linear4bit
    check(exits(ij.check_quantized_scope, scope_model) == 2, "a quantised vision-tower layer was not noticed")
    mods[vision].__class__ = torch.nn.Linear
    mods[lm[0]].__class__ = torch.nn.Linear
    check(exits(ij.check_quantized_scope, scope_model) == 2, "an unquantised language-model layer was not noticed")
    del scope_model, mods

    # token_type_ids always mark exactly the image tokens, whatever the processor returns
    enc = ij.encode_job(proc, jobs[0], d["bundle"], ids)
    check(torch.equal(enc["token_type_ids"], (enc["input_ids"] == ids["image"]).long()) and int(enc["token_type_ids"].sum()) == 256,
          "token_type_ids do not mark the image tokens")
    check(int(ij.encode_job(proc, jobs[3], d["bundle"], ids)["token_type_ids"].sum()) == 0, "text-only job: token_type_ids")

    class NoTypeIds:                                                           # a processor version that returns no token_type_ids
        def __init__(self, inner, wrong=False):
            self.inner, self.wrong, self.tokenizer = inner, wrong, inner.tokenizer

        def apply_chat_template(self, *a_, **k_):
            out = dict(self.inner.apply_chat_template(*a_, **k_))
            tt = out.pop("token_type_ids")
            if self.wrong:
                out["token_type_ids"] = torch.zeros_like(tt)
            return out

    built = ij.encode_job(NoTypeIds(proc), jobs[0], d["bundle"], ids)
    check(torch.equal(built["token_type_ids"], enc["token_type_ids"]), "token_type_ids were not built when the processor returns none")
    check(exits(ij.encode_job, NoTypeIds(proc, wrong=True), jobs[0], d["bundle"], ids) == 2, "wrong processor token_type_ids were accepted")
    check(ij.token_type_ids_source(proc).startswith("processor") and ij.token_type_ids_source(NoTypeIds(proc)).startswith("built"),
          "token_type_ids_source")
    notes.append("dev jobs are refused with a non-final, debug or meta-less adapter (smoke jobs are not; the override is recorded); "
                 f"4-bit scope: {rule}, and a quantised vision layer is caught on the loaded model; token_type_ids are built from "
                 "the image-token positions when the processor returns none")

    old_argv, old_env = sys.argv, os.environ.get("INFER_JOBS_ARGS")
    try:                                                  # a notebook kernel starts the script as `launcher -f kernel.json`
        sys.argv = ["ipykernel_launcher.py", "-f", "/tmp/kernel-1.json"]
        os.environ["INFER_JOBS_ARGS"] = "--jobs x.jsonl --batch-size 2 --tag 'a b'"
        a_ = ij.parse_args()
        check((a_.jobs, a_.batch_size, a_.tag) == ("x.jsonl", 2, "a b"), "INFER_JOBS_ARGS not used in a kernel")
        sys.argv = ["ipykernel_launcher.py", "--jobs", "y.jsonl"]             # header-injected arguments are kept
        check(ij.parse_args().jobs == "y.jsonl", "injected arguments were dropped")
    finally:
        sys.argv = old_argv
        os.environ.pop("INFER_JOBS_ARGS", None) if old_env is None else os.environ.__setitem__("INFER_JOBS_ARGS", old_env)

    import builtins
    source = open(ij.__file__).read()
    check(ij.script_path() == os.path.abspath(ij.__file__), "script_path with a file")
    saved = ij.__dict__.pop("__file__")
    try:
        check(ij.script_path() is None, "script_path without a file or IPython should be None")
        builtins.get_ipython = lambda: types.SimpleNamespace(user_ns={"_ih": ["", source]})
        real_tmp, ij.tempfile.gettempdir = ij.tempfile.gettempdir, (lambda: work)      # keep the temporary script inside the test folder
        try:
            tmp_script = ij.script_path()
        finally:
            ij.tempfile.gettempdir = real_tmp
        check(tmp_script and open(tmp_script).read() == source, "script_path did not write the running cell")
        os.remove(tmp_script)
    finally:
        ij.__file__ = saved
        if hasattr(builtins, "get_ipython"):
            del builtins.get_ipython
    notes.append("precision follows the adapter's step4_meta.json (4-bit/float32 or bf16), else bf16 on native-bf16 GPUs and 4-bit on a T4; "
                 "adapter fingerprint, nested dataset folders, kernel-style argv and the worker script fallback work")
    return "; ".join(notes)


TESTS = [("reference", t_reference), ("build", t_build), ("jobs", t_jobs), ("tokens", t_tokens), ("leak", t_leak),
         ("validate", t_validate), ("import", t_import), ("import_refusals", t_import_refusals), ("pinned", t_pinned),
         ("sampler", t_sampler), ("tiny", t_tiny)]
NEEDS = {"jobs": ["reference", "build"], "tokens": ["reference", "build"], "leak": ["build"], "validate": ["build"],
         "import": ["reference", "build"], "import_refusals": ["reference", "build"], "tiny": ["build"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=[n for n, _ in TESTS], help="run these tests (plus what they need)")
    ap.add_argument("--no-tiny-model", action="store_true", help="skip the tiny-model end-to-end test")
    ap.add_argument("--keep", action="store_true", help=f"keep this run's folder under {TMP_BASE}")
    args = ap.parse_args()
    wanted = set(args.only or [n for n, _ in TESTS])
    for n in list(wanted):
        wanted |= set(NEEDS.get(n, []))
    if args.no_tiny_model:
        wanted.discard("tiny")
    import torch
    import transformers
    print(f"python {sys.version.split()[0]}, transformers {transformers.__version__}, torch {torch.__version__} (CPU), "
          f"processor files {MODEL_DIR}")
    setup()
    results = []
    for name, fn in TESTS:
        if name not in wanted:
            continue
        t0 = time.time()
        try:
            note = fn()
            results.append((name, True))
            print(f"PASS {name} ({time.time() - t0:.0f}s): {note}", flush=True)
        except Exception as e:                                           # noqa: BLE001 - report and continue
            results.append((name, False))
            print(f"FAIL {name} ({time.time() - t0:.0f}s): {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    if args.keep:
        print(f"kept {TMP}")
    else:
        shutil.rmtree(TMP, ignore_errors=True)        # this run's own folder only
    failed = [n for n, ok in results if not ok]
    print(f"{len(results) - len(failed)}/{len(results)} tests passed" + (f"; FAILED: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
