#!/usr/bin/env python3
r"""Batched, resumable, seeded generation over a job file: the Kaggle side of the MHIST step-4 evaluation.

It runs inside a PRIVATE Kaggle notebook (GPU T4 x2 or P100). It needs no network: the model weights, the
data bundle (gridded tiles), the job files and, optionally, a LoRA adapter are attached as PRIVATE datasets
under /kaggle/input/. The same file runs on CPU for the local tests (kaggle_ft/test_jobs_local.py).

    python infer_jobs.py --print-env
    python infer_jobs.py --jobs /kaggle/input/<jobs>/dev__cte_p1__none__tier1.jsonl --bundle-dir /kaggle/input/<bundle>
        --model-dir /kaggle/input/<weights> [--adapter-dir /kaggle/input/<adapter>]

Paths can also come from the environment (MEDGEMMA_DIR, LORA_ADAPTER_DIR, MHIST_BUNDLE_DIR), and the whole
argument list from INFER_JOBS_ARGS when a notebook kernel starts the script without arguments.

JOB FILE (one JSON object per line; it carries NO labels and NO metadata -- those stay in the local sidecar)
    {"job_id": str, "parts": [{"type": "text", "text": ...} | {"type": "image", "file": "gridded/MHIST_xxx.png",
     "sha256": <optional, checked>} ...], "tier": "1" | "2", "seed": int, "max_new_tokens": 1500}
    Image files are relative to the data bundle (--bundle-dir: the attached dataset that holds gridded/).

PROMPT
    One user turn, parts in order, no system prompt, built by the processor's own apply_chat_template with
    add_generation_prompt. That is the official format: text parts are stripped and concatenated and each image
    becomes "\n\n<start_of_image>" + 256 image tokens + "<end_of_image>\n\n". Every job must show exactly
    256 image tokens per image, or the run stops (the image was not ingested).

DECODING
    tier 1: sampling at temperature 1.0, top_k 64, top_p 0.95.     tier 2: greedy.
    Thinking is off in both: bad_words_ids=[[100]] bans <unused94>, the token that opens a trace. If that token
    is generated anyway the run stops. The answer is decoded WITHOUT skipping special tokens, so <unused95> and
    <end_of_turn> stay visible in raw_response.

SEEDING AND BATCH DEPENDENCE
    --sampler per-job (default). Each job owns a private CPU random generator seeded with the job's seed. At
    each step the row's logits go through transformers' own TopK/TopP warpers one row at a time, and one uniform
    number from that job's generator picks the token by inverse CDF over the kept tokens in ascending token-id
    order. The random stream of a job therefore depends on its seed alone: not on the order of the jobs, not on
    the shard or worker it ran in, and not on the other jobs in its batch.
    --sampler hf uses stock generate(do_sample=True) after torch.manual_seed(seed). It is allowed at batch size
    1 only, because a stock batch shares one random stream.
    RESIDUAL BATCH DEPENDENCE (the reason the default batch size is 1). With batch size > 1 the shorter prompts
    are left-padded and the forward pass runs different batched kernels, so the logits differ at float level
    (most in bfloat16). A sampled token flips when its uniform number lands within that noise of a CDF boundary,
    a greedy token flips on a near-tie, and after one flip the rest of that answer differs. Batch size 1 has
    neither effect. Use --batch-size N only after measuring it on this GPU:
        python infer_jobs.py --jobs smoke.jsonl ... --batch-size 1 --tag bs1
        python infer_jobs.py --jobs smoke.jsonl ... --batch-size 4 --tag bs4
        python infer_jobs.py --compare <bs1 file> <bs4 file>
    Batch size, sampler, precision and library versions are written into every output line, and a resumed run
    refuses to continue an output file that was started with different settings.
    Results are reproducible on the same GPU model and library versions. Another GPU model can change logits at
    float level, with the same consequences as above.

PRECISION
    --precision auto loads the base model the way the adapter was trained (best_adapter/step4_meta.json written
    by train_lora.py). Without that file it follows train_lora.py's rule: bf16 weights on a GPU with native bf16
    (compute capability >= 8), otherwise (Kaggle's T4, P100) 4-bit nf4 with float32 arithmetic. fp16 is never
    used (Gemma 3 overflows in it). The adapter is loaded on top of the base weights and is NOT merged.
    4-bit quantises the language model's linear layers only, exactly as train_lora.py does: the SigLIP vision
    tower, the projector, the embeddings and lm_head stay unquantised. That is checked on the loaded model, and
    an adapter trained with another 4-bit scope is refused.

ADAPTER
    Dev job files (job ids starting with dev__) are run only with the FINAL adapter of a protocol run: the
    best_adapter/ folder of a finished train_lora.py run, whose step4_meta.json says "protocol_run": true and
    "final": true and records the adapter's sha256. An epoch_N checkpoint, the best-so-far adapter of a paused
    run, a debug run's adapter or a folder without step4_meta.json is refused (--allow-non-final-adapter overrides
    and is written into every output line; import_results.py then refuses the file). Smoke job files (pool
    tiles, never scored) accept any adapter.

OUTPUT (/kaggle/working/<job file stem>__<tag>.jsonl, one line per job, appended and fsynced as jobs finish)
    job_id, raw_response, n_prompt_tokens, n_new_tokens, finish_reason ("stop" if <eos>/<end_of_turn> was
    generated, else "length"), seconds, tier, seed, max_new_tokens, n_images, prompt_ids_sha256, model_revision,
    weights_verified, images_verified, adapter_sha256, dtype, quantization, batch_size, sampler, run_signature,
    jobs_sha256, run.
    <out>.status.json says complete / incomplete / failed.

RESUME
    Same session: re-run the same command. Job ids already in the output file (or in a worker's shard file) are
    skipped. A line cut off by a crash is dropped and that job is redone.
    New Kaggle session (/kaggle/working starts empty): attach the earlier notebook's output and pass
    --resume-from <that folder or file>; its finished answers are copied first.
    Kaggle discards /kaggle/working when a notebook exits non-zero. So on Kaggle (--exit-zero auto) a run that
    stops early with answers already written exits 0; the status file and the last log line say it is not
    complete, and import_results.py refuses an incomplete file.

WORKERS
    --workers N starts N processes, one per GPU (CUDA_VISIBLE_DEVICES=i), each taking the jobs whose index
    % N == i and writing <out>.shard<i>of<N>. When they finish, the shard files are merged into <out> in job
    order. With per-job seeding and batch size 1 the result does not depend on N. Default: one per visible GPU.

REQUIREMENTS (nothing is downloaded; a missing or too-old package stops the run with a message naming it)
    torch >= 2.2, transformers >= 4.50.0 (first release with Gemma3ForConditionalGeneration; the pinned weights
    were saved with 4.57.1), Pillow; accelerate >= 0.26 on GPU; peft >= 0.13 only with --adapter-dir;
    bitsandbytes >= 0.43 only for 4-bit.
"""

import argparse
import glob
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import traceback

PINNED_MODEL = "google/medgemma-1.5-4b-it"
PINNED_REVISION = "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"
# Published hashes of that revision: sha256 for the LFS files, git blob sha1 for the small ones.
PINNED_SHA256 = {
    "model-00001-of-00002.safetensors": "5e4c75b0ef1fb009caee567ab244f9e354e915fda748d1b76179cc453a39b4b5",
    "model-00002-of-00002.safetensors": "958e39df78c35ddb812fbcb8b5e7f46e07f1b153c1589f2d3f0b41c3b0748d30",
    "tokenizer.json": "7d4046bf0505a327dd5a0abbb427ecd4fc82f99c2ceaa170bc61ecde12809b0c",
}
PINNED_GIT_BLOB = {
    "config.json": "d3842cc948343b17846a93ec9a91d5b325dc7f64",
    "chat_template.jinja": "1117055ab8e8c90e1b200be00cddb78943616d9e",
    "generation_config.json": "19b5c70598644fcc255c7be7099b1af631e956ef",
    "preprocessor_config.json": "cbd4f0cd77e39566f11921f9995bb3a4d008a83e",
    "processor_config.json": "453c7966d4b5d0b4a317c585989f64c58c2a6bf0",
    "special_tokens_map.json": "1a6193244714d3d78be48666cb02cdbfac62ad86",
    "tokenizer_config.json": "ad0c34f8dd76c7138b35b3973f3b4a422d5136a9",
    "added_tokens.json": "e17bde03d42feda32d1abfca6d3b598b9a020df7",
    "model.safetensors.index.json": "4b95241f208f06d324d17c9675568ec58dafd9fb",
}
WEIGHT_FILES = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")

THINK_OPEN, THINK_OPEN_ID = "<unused94>", 100     # opens a thinking trace; banned in every call
THINK_CLOSE, THINK_CLOSE_ID = "<unused95>", 101
IMAGE_TOKENS_PER_IMAGE = 256
TIERS = {"1": {"do_sample": True, "temperature": 1.0, "top_k": 64, "top_p": 0.95},
         "2": {"do_sample": False}}
JOB_KEYS = ("job_id", "parts", "tier", "seed", "max_new_tokens")
MIN_VERSIONS = {"torch": "2.2.0", "transformers": "4.50.0", "accelerate": "0.26.0", "peft": "0.13.0",
                "bitsandbytes": "0.43.0"}                       # the same floor as train_lora.py
SAVED_WITH_TRANSFORMERS = "4.57.1"
KAGGLE_WORKING = "/kaggle/working"
EXIT_TIME_BUDGET = 3
DEV_JOB_PREFIX = "dev__"                        # job ids of the dev evaluation (build_dev_jobs.py: <tiles>__<config>__...)
# The same two constants as in train_lora.py (the two files are pushed separately, so they are repeated here).
# One skip name per module-tree layout and per matching rule of the transformers releases in use: with
# "vision_tower" alone, transformers 5.18 quantises the whole vision tower. The loaded model is checked.
BNB_SKIP_MODULES = ("vision_tower", "model.vision_tower", "multi_modal_projector", "model.multi_modal_projector",
                    "lm_head", "language_model.lm_head")
QUANTIZED_SCOPE = "language-model linear layers only; vision_tower, multi_modal_projector, embeddings and lm_head not quantised"
LM_LINEAR_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# Test hook only (kaggle_ft/test_jobs_local.py): extra logits processors run before the sampler. Empty in use.
_TEST_LOGITS_PROCESSORS = []


def log(msg, prefix=""):
    print(f"{prefix}{msg}", flush=True)


def die(msg, code=2):
    print(f"STOP: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def vtuple(version):
    return tuple(int(x) for x in re.findall(r"\d+", str(version))[:3])


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(path):
    h = hashlib.sha1()
    h.update(b"blob %d\0" % os.path.getsize(path))
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1 << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def ids_sha256(ids):
    """Hash of a token-id sequence; the local sidecar holds the expected value for every job."""
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------- job files

def validate_job(job, where="job"):
    if not isinstance(job, dict) or set(job) != set(JOB_KEYS):
        raise ValueError(f"{where}: keys must be exactly {list(JOB_KEYS)}, got {sorted(job) if isinstance(job, dict) else type(job).__name__}"
                         " (labels and metadata belong in the local sidecar, never in a job file)")
    if not isinstance(job["job_id"], str) or not job["job_id"]:
        raise ValueError(f"{where}: job_id must be a non-empty string")
    if job["tier"] not in TIERS:
        raise ValueError(f"{where}: tier must be one of {sorted(TIERS)} (strings), got {job['tier']!r}")
    for key, low in (("seed", 0), ("max_new_tokens", 1)):
        if not isinstance(job[key], int) or isinstance(job[key], bool) or not low <= job[key] < 2 ** 63:
            raise ValueError(f"{where}: {key} must be an integer >= {low}, got {job[key]!r}")
    if not isinstance(job["parts"], list) or not job["parts"]:
        raise ValueError(f"{where}: parts must be a non-empty list")
    for n, part in enumerate(job["parts"]):
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "text":
            ok = set(part) == {"type", "text"} and isinstance(part["text"], str)
        elif kind == "image":
            f = part.get("file")
            ok = (set(part) <= {"type", "file", "sha256"} and isinstance(f, str) and f != ""
                  and not os.path.isabs(f) and ".." not in f.replace("\\", "/").split("/"))
        else:
            ok = False
        if not ok:
            raise ValueError(f"{where}: parts[{n}] must be {{'type':'text','text':str}} or "
                             f"{{'type':'image','file':<relative path>[,'sha256':hex]}}, got {part!r}")
    return job


def read_jobs(path):
    """Returns (jobs, sha256 of the job file). Stops on a malformed line or a repeated job_id."""
    raw = open(path, "rb").read()
    jobs, seen = [], set()
    for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            job = validate_job(json.loads(line), f"{os.path.basename(path)} line {n}")
        except (ValueError, json.JSONDecodeError) as e:
            die(str(e))
        if job["job_id"] in seen:
            die(f"{path} line {n}: job_id {job['job_id']!r} appears twice")
        seen.add(job["job_id"])
        jobs.append(job)
    if not jobs:
        die(f"{path} has no jobs")
    return jobs, hashlib.sha256(raw).hexdigest()


def n_images_of(job):
    return sum(1 for p in job["parts"] if p["type"] == "image")


def resolve_image(bundle_dir, rel):
    path = os.path.join(bundle_dir, rel)
    if os.path.isfile(path):
        return path
    hits = sorted(glob.glob(os.path.join(glob.escape(bundle_dir), "*", rel)))   # dataset mounted one level deeper
    if len(hits) == 1:
        return hits[0]
    die(f"image {rel!r} not found under {bundle_dir} ({len(hits)} matches one level down). "
        "Attach the data bundle and pass --bundle-dir.")


def default_bundle_dir(jobs_path):
    """The folder that holds gridded/: next to the job file, one level up, or the one attached Kaggle dataset that has it."""
    d = os.path.dirname(os.path.abspath(jobs_path))
    for cand in (d, os.path.dirname(d)):
        if os.path.isdir(os.path.join(cand, "gridded")):
            return cand
    hits = sorted({os.path.dirname(p) for pat in ("*", os.path.join("*", "*"))
                   for p in glob.glob(os.path.join("/kaggle/input", pat, "gridded")) if os.path.isdir(p)})
    if len(hits) == 1:
        return hits[0]
    die(f"cannot find the data bundle (a folder with gridded/): {len(hits)} candidates under /kaggle/input "
        f"{hits[:4]}. Attach the bundle dataset and pass --bundle-dir or set MHIST_BUNDLE_DIR.")


# ------------------------------------------------------------------------------------------- output files

def read_output(path, repair=False):
    """Records of an output file. A crash can cut the last line: it is ignored, and removed when repair=True."""
    if not os.path.exists(path):
        return []
    raw = open(path, "rb").read()
    recs, good_end, pos = [], 0, 0
    for line in raw.splitlines(keepends=True):
        pos += len(line)
        if not line.strip():
            good_end = pos
            continue
        try:
            rec = json.loads(line)
            if not line.endswith(b"\n") or not isinstance(rec, dict) or "job_id" not in rec:
                raise ValueError("incomplete")
        except ValueError:
            if pos != len(raw):
                die(f"{path}: unreadable line before the end of the file; move it aside and re-run")
            if repair:
                with open(path, "r+b") as fh:
                    fh.truncate(good_end)
                log(f"dropped a cut-off last line from {path}; that job will be redone")
            break
        recs.append(rec)
        good_end = pos
    return recs


def shard_files(out):
    return sorted(glob.glob(glob.escape(out) + ".shard*of*"))


def merge_shards(out, job_order):
    """Fold worker shard files into `out` in job order (atomic), then delete them."""
    shards = shard_files(out)
    if not shards:
        return
    merged = {}
    for path in [out] + shards:
        for rec in read_output(path):
            old = merged.get(rec["job_id"])
            if old is not None and old != rec:
                die(f"job {rec['job_id']!r} has two different results ({path} and an earlier file). "
                    "Nothing merged; remove the wrong file by hand.")
            merged[rec["job_id"]] = rec
    order = {j: n for n, j in enumerate(job_order)}
    recs = sorted(merged.values(), key=lambda r: order.get(r["job_id"], len(order)))
    tmp = out + ".merging"
    with open(tmp, "w") as fh:
        for rec in recs:
            fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)
    for path in shards:
        os.remove(path)
    log(f"merged {len(shards)} shard file(s) into {out} ({len(recs)} jobs)")


def earlier_answers(sources, out):
    """Answer files of an earlier session: the given files, and files named like `out` (or its shards) in the given folders."""
    base, paths = os.path.basename(out), []
    for src in sources or []:
        if os.path.isfile(src):
            paths.append(src)
        elif os.path.isdir(src):
            for root, _, files in os.walk(src):
                paths += [os.path.join(root, f) for f in files if f == base or f.startswith(base + ".shard")]
        else:
            die(f"--resume-from {src} does not exist")
    return sorted(set(os.path.abspath(p) for p in paths) - {os.path.abspath(out)})


# ------------------------------------------------------------------------------------------- environment

def check_env(need_peft=False, need_bnb=False, need_accelerate=False):
    """Versions of what is installed. Stops with a clear message if something required is missing or too old."""
    v, problems = {"python": sys.version.split()[0]}, []
    for mod, name, required in (("torch", "torch", True), ("transformers", "transformers", True),
                                ("PIL", "pillow", True), ("tokenizers", "tokenizers", False),
                                ("accelerate", "accelerate", need_accelerate), ("peft", "peft", need_peft),
                                ("bitsandbytes", "bitsandbytes", need_bnb), ("safetensors", "safetensors", False),
                                ("torchvision", "torchvision", False)):
        try:
            v[name] = str(getattr(__import__(mod), "__version__", "?"))
        except Exception as e:   # noqa: BLE001 - any import failure means "not usable here"
            v[name] = None
            if required:
                problems.append(f"{name} is required here but cannot be imported ({type(e).__name__}: {str(e)[:120]})")
            continue
        if required and name in MIN_VERSIONS and vtuple(v[name]) < vtuple(MIN_VERSIONS[name]):
            problems.append(f"{name} {v[name]} is too old, need >= {MIN_VERSIONS[name]}")
    if problems:
        die("; ".join(problems) + ". Internet is assumed OFF, so nothing was installed. Attach wheels as a private "
            "dataset and pip install --no-index them before this script (kaggle_push.py --pip-install), or use an "
            "image that ships the packages.")
    if vtuple(v["transformers"]) < vtuple(SAVED_WITH_TRANSFORMERS):
        log(f"note: transformers {v['transformers']} is older than {SAVED_WITH_TRANSFORMERS}, the release the pinned weights were saved with")
    return v


def gpu_info():
    import torch
    if not torch.cuda.is_available():
        return {"cuda": False, "n_gpus": 0, "gpus": []}
    caps = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
    return {"cuda": True, "n_gpus": torch.cuda.device_count(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "compute_capability": [f"{a}.{b}" for a, b in caps],
            "memory_gb": [round(torch.cuda.get_device_properties(i).total_memory / 2 ** 30, 1) for i in range(torch.cuda.device_count())],
            "native_bf16": all(a >= 8 for a, _ in caps)}


def find_dir_with(root, filename, what):
    """`root` itself, or the single folder up to two levels below it that holds `filename` (Kaggle nests datasets)."""
    if not root:
        die(f"no {what} given")
    if os.path.isfile(os.path.join(root, filename)):
        return os.path.abspath(root)
    hits = sorted({os.path.dirname(p) for pat in ("*", os.path.join("*", "*"))
                   for p in glob.glob(os.path.join(glob.escape(root), pat, filename))})
    if len(hits) == 1:
        return os.path.abspath(hits[0])
    die(f"{what}: {filename} not found in {root} ({len(hits)} candidates below it: {hits[:4]})")


def fingerprint_model(model_dir, cache_path, hash_weights=True):
    """Hash the attached weights and compare them with the pinned revision. Returns a dict for the output lines."""
    info = {"model_dir": model_dir, "sha256": {}, "git_blob": {}, "mismatch": [], "missing": []}
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path))
        except ValueError:
            cache = {}
    for name, want in sorted(PINNED_GIT_BLOB.items()):
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path):
            if name == "config.json":
                info["missing"].append(name)
            continue
        got = git_blob_sha1(path)
        info["git_blob"][name] = got
        if got != want:
            info["mismatch"].append(name)
    for name, want in sorted(PINNED_SHA256.items()):
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path):
            if name in WEIGHT_FILES:
                info["missing"].append(name)
            continue
        if not hash_weights and name in WEIGHT_FILES:
            continue
        st = os.stat(path)
        key = f"{os.path.realpath(path)}|{st.st_size}|{int(st.st_mtime)}"
        if key not in cache:
            log(f"hashing {name} ({st.st_size / 2 ** 30:.2f} GiB) ...")
            cache[key] = sha256_file(path)
            if cache_path:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                tmp = f"{cache_path}.{os.getpid()}.tmp"
                json.dump(cache, open(tmp, "w"))
                os.replace(tmp, cache_path)
        info["sha256"][name] = cache[key]
        if cache[key] != want:
            info["mismatch"].append(name)
    hashed_all = all(n in info["sha256"] for n in WEIGHT_FILES)
    if info["mismatch"] or info["missing"]:
        info["verified"] = False
    elif hashed_all:
        info["verified"] = True
    else:
        info["verified"] = None     # weight hashing was skipped
    return info


def fingerprint_adapter(adapter_dir):
    """Hashes of the adapter, its LoRA settings, and what train_lora.py recorded about how it was trained."""
    if not adapter_dir:
        return None
    weights = [f for f in ("adapter_model.safetensors", "adapter_model.bin") if os.path.isfile(os.path.join(adapter_dir, f))]
    if not weights:
        die(f"{adapter_dir} has adapter_config.json but no adapter_model.safetensors / adapter_model.bin")
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    meta_path, meta = os.path.join(adapter_dir, "step4_meta.json"), None
    if os.path.isfile(meta_path):
        meta = json.load(open(meta_path))
    return {"adapter_dir": adapter_dir, "weights_file": weights[0],
            "sha256": sha256_file(os.path.join(adapter_dir, weights[0])),
            "config_sha256": sha256_file(os.path.join(adapter_dir, "adapter_config.json")),
            "r": cfg.get("r"), "lora_alpha": cfg.get("lora_alpha"), "peft_type": cfg.get("peft_type"),
            "base_model_name_or_path": cfg.get("base_model_name_or_path"),
            "trained": None if meta is None else {"epoch": meta.get("epoch"), "precision": meta.get("precision"),
                                                  "base_model": meta.get("base_model"),
                                                  "image_processor_class": meta.get("image_processor_class"),
                                                  "protocol_run": meta.get("protocol_run"),
                                                  "final": meta.get("final"), "chosen_epoch": meta.get("chosen_epoch"),
                                                  "epochs_completed": meta.get("epochs_completed"),
                                                  "epochs_planned": meta.get("epochs_planned"),
                                                  "adapter_model_sha256": meta.get("adapter_model_sha256"),
                                                  "target_variant": meta.get("target_variant")}}


def adapter_gate(adapter, jobs, allow_non_final=False):
    """Checkpoint selection uses the validation slice only (PLAN, step-4 addendum). So the dev tiles are shown to
    ONE adapter: the final, validation-selected best_adapter/ of a finished protocol run. Anything else (an
    epoch_N checkpoint, the best-so-far adapter of a paused run, a debug run, a folder without step4_meta.json)
    stops here, before any dev tile is read. Returns the list of reasons (empty = the adapter is the final one)."""
    if not adapter:
        return []
    t = adapter.get("trained") or {}
    recorded = t.get("adapter_model_sha256")
    if recorded and recorded != adapter["sha256"]:
        die(f"{adapter['adapter_dir']}: step4_meta.json records adapter sha256 {recorded[:12]}..., but the weights file "
            f"hashes to {adapter['sha256'][:12]}...: the meta file and the weights are not from the same folder")
    reasons = []
    if adapter.get("trained") is None:
        reasons.append("it has no step4_meta.json (not an unchanged train_lora.py output folder)")
    else:
        if t.get("protocol_run") is not True:
            reasons.append(f"protocol_run is {t.get('protocol_run')!r} (a debug or smoke run of train_lora.py)")
        if t.get("final") is not True:
            reasons.append(f"final is {t.get('final')!r}: an epoch checkpoint or the best-so-far adapter of an unfinished run"
                           f" (epoch {t.get('epoch')!r}, {t.get('epochs_completed')!r} of {t.get('epochs_planned')!r} epochs validated)")
        elif not recorded:
            reasons.append("it is marked final but records no adapter_model_sha256")
    n_dev = sum(1 for j in jobs if str(j["job_id"]).startswith(DEV_JOB_PREFIX))
    if reasons and n_dev:
        msg = (f"{n_dev} dev jobs, but the adapter in {adapter['adapter_dir']} is not the final adapter of a protocol run: "
               + "; ".join(reasons) + ". Dev tiles are evaluated once, with best_adapter/ of the finished run "
               "(train_summary.json status complete).")
        if not allow_non_final:
            die(msg + " Nothing was run. (--allow-non-final-adapter overrides; import_results.py refuses such answers.)")
        log("WARNING: " + msg + " Continuing because of --allow-non-final-adapter; every output line records it.")
    return reasons


def resolve_precision(args, adapter, gpus):
    """How the base model is loaded. Returns {mode, weights_dtype, compute_dtype, quantization, reason, as_trained}."""
    trained = ((adapter or {}).get("trained") or {}).get("precision") or None
    mode, compute, reason = args.precision, args.compute_dtype, f"--precision {args.precision}"
    if args.device == "cpu":
        if mode not in ("auto", "fp32"):
            die("--device cpu only supports --precision fp32 (local tests)")
        mode, reason = "fp32", "CPU run (tests only)"
    elif mode == "auto":
        if trained and trained.get("mode") in ("bf16", "4bit", "float32"):
            mode = {"float32": "fp32"}.get(trained["mode"], trained["mode"])
            if compute == "auto" and trained.get("compute_dtype") in ("bfloat16", "float32"):
                compute = {"bfloat16": "bf16", "float32": "fp32"}[trained["compute_dtype"]]
            reason = "auto: as the adapter was trained (step4_meta.json)"
        elif gpus.get("native_bf16"):
            mode, reason = "bf16", "auto: GPU with native bf16"
        else:
            mode, reason = "4bit", ("auto: no native bf16 on " + ", ".join(gpus.get("gpus") or ["this GPU"])
                                    + ", so 4-bit nf4 with float32 arithmetic (train_lora.py's rule)")
    if mode == "bf16":
        out = {"mode": "bf16", "weights_dtype": "bfloat16", "compute_dtype": "bfloat16", "quantization": None}
        if args.device != "cpu" and not gpus.get("native_bf16"):
            reason += " (bf16 is EMULATED on this GPU: correct but possibly slow)"
    elif mode == "fp32":
        out = {"mode": "fp32", "weights_dtype": "float32", "compute_dtype": "float32", "quantization": None}
    else:
        if compute == "auto":
            compute = "bf16" if gpus.get("native_bf16") else "fp32"
        cd = {"bf16": "bfloat16", "fp32": "float32"}[compute]
        out = {"mode": "4bit", "weights_dtype": "4-bit nf4 (double quantisation); unquantised modules " + cd,
               "compute_dtype": cd, "quantization": "bnb-4bit-nf4-double-quant",
               "quantized_scope": QUANTIZED_SCOPE, "vision_tower": cd + ", not quantised"}
    out.setdefault("quantized_scope", None)
    out.setdefault("vision_tower", out["weights_dtype"] + ", not quantised")
    out["reason"] = reason
    if trained:
        t_mode = {"float32": "fp32"}.get(trained.get("mode"), trained.get("mode"))
        out["as_trained"] = bool(t_mode == out["mode"] and (out["mode"] != "4bit" or trained.get("compute_dtype") == out["compute_dtype"]))
        if out["mode"] == "4bit" and t_mode == "4bit" and trained.get("quantized_scope") != QUANTIZED_SCOPE:
            die(f"the adapter was trained with another 4-bit scope ({trained.get('quantized_scope')!r}; an older train_lora.py "
                f"also quantised the vision tower). This script quantises: {QUANTIZED_SCOPE}. Training and evaluation "
                "must load the base model the same way: retrain with the current train_lora.py.")
    else:
        out["as_trained"] = None
    return out


def check_quantized_scope(model):
    """After a 4-bit load: exactly the language model's linear layers (7 per block) are Linear4bit, nothing in the
    vision tower, the projector or lm_head. Same check as in train_lora.py. Returns the number of 4-bit modules."""
    names = [n for n, m in model.named_modules() if type(m).__name__ == "Linear4bit"]
    n_layers = model.config.text_config.num_hidden_layers

    def inside_lm(name):
        parts = name.split(".")
        return ("language_model" in parts and "vision_tower" not in parts and "multi_modal_projector" not in parts
                and parts[-1] in LM_LINEAR_LEAVES)

    outside = [n for n in names if not inside_lm(n)]
    if outside or len(names) != len(LM_LINEAR_LEAVES) * n_layers:
        import transformers
        die(f"4-bit scope is wrong with transformers {transformers.__version__}: {len(names)} Linear4bit modules, expected "
            f"{len(LM_LINEAR_LEAVES)} x {n_layers}, all inside the language model; {len(outside)} lie outside it, e.g. "
            f"{outside[:3]}. BNB_SKIP_MODULES does not match this version's module names.")
    return len(names)


# ------------------------------------------------------------------------------------------------ model

def load_processor(model_dir, image_backend="default"):
    """The model's own processor, loaded exactly as train_lora.py loads it ('default' = AutoProcessor's choice)."""
    import transformers
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    if image_backend == "pil":
        from transformers import AutoImageProcessor
        kw = {"backend": "pil"} if vtuple(transformers.__version__) >= (5,) else {"use_fast": False}
        processor.image_processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, **kw)
    tok = processor.tokenizer
    for token, want in ((THINK_OPEN, THINK_OPEN_ID), (THINK_CLOSE, THINK_CLOSE_ID)):
        got = tok.convert_tokens_to_ids(token)
        if got != want:
            die(f"{token} has id {got} in this tokenizer, expected {want}: not the pinned MedGemma tokenizer")
    return processor


def special_ids(processor):
    tok = processor.tokenizer
    eos = sorted({int(tok.eos_token_id), int(tok.convert_tokens_to_ids("<end_of_turn>"))})
    return {"eos": eos, "pad": int(tok.pad_token_id), "image": int(tok.convert_tokens_to_ids("<image_soft_token>"))}


def load_model(args, versions, precision):
    import torch
    from transformers import AutoModelForImageTextToText
    dtype = getattr(torch, precision["compute_dtype"])
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        die("no CUDA device is visible. In the Kaggle notebook choose Settings > Accelerator > GPU T4 x2 "
            "(or pass --device cpu for a dry test).")
    if device == "cuda":
        device = "cuda:0"
    kw = {"local_files_only": True, "attn_implementation": args.attn_implementation,
          ("dtype" if vtuple(versions["transformers"]) >= (5,) else "torch_dtype"): dtype}
    if precision["mode"] == "4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                                       bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                                                       llm_int8_skip_modules=list(BNB_SKIP_MODULES))
        kw["device_map"] = "auto" if device == "split" else {"": device}
    elif device == "split":
        kw["device_map"] = "auto"
    elif device != "cpu" and versions.get("accelerate"):
        kw["device_map"] = {"": device}
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, **kw)
    if "device_map" not in kw:
        model.to(device)
    if type(model).__name__ != "Gemma3ForConditionalGeneration":
        die(f"loaded a {type(model).__name__}, expected Gemma3ForConditionalGeneration")
    if precision["mode"] == "4bit":
        log(f"4-bit scope checked: {check_quantized_scope(model)} language-model linear layers quantised, vision tower untouched")
    if args.adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir, is_trainable=False)
    model.eval()
    first = next(model.parameters())
    log(f"model loaded in {time.time() - t0:.0f}s: {type(model).__name__}, {precision['weights_dtype']}, "
        f"arithmetic {precision['compute_dtype']}, attention {args.attn_implementation}, first parameter on {first.device}")
    return model, first.device, dtype


def encode_job(processor, job, bundle_dir, ids=None, verify_images=True):
    """Official chat format for one job. Returns the processor's tensors (batch of 1) plus bookkeeping."""
    from PIL import Image
    ids = ids or special_ids(processor)
    content = []
    for part in job["parts"]:
        if part["type"] == "text":
            content.append({"type": "text", "text": part["text"]})
            continue
        raw = open(resolve_image(bundle_dir, part["file"]), "rb").read()
        if verify_images and part.get("sha256") and hashlib.sha256(raw).hexdigest() != part["sha256"]:
            die(f"{part['file']} does not match the sha256 in job {job['job_id']}: the bundle and the job file "
                "come from different builds")
        content.append({"type": "image", "image": Image.open(io.BytesIO(raw)).convert("RGB")})
    enc = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                        tokenize=True, return_dict=True, return_tensors="pt")
    enc = dict(enc)
    # token_type_ids mark the image tokens; in Gemma 3 they switch each image block to bidirectional attention.
    # train_lora.py always builds them from the image-token positions. Do the same here whatever this
    # transformers version's processor returns, so that evaluation cannot silently run causal attention over
    # the image tokens while training did not.
    import torch
    want = (enc["input_ids"] == ids["image"]).long()
    if "token_type_ids" not in enc:
        enc["token_type_ids"] = want
    elif not torch.equal(enc["token_type_ids"] != 0, want != 0):
        die(f"job {job['job_id']}: the processor's token_type_ids do not mark exactly the image tokens")
    row = enc["input_ids"][0].tolist()
    n_images = n_images_of(job)
    if row.count(ids["image"]) != IMAGE_TOKENS_PER_IMAGE * n_images:
        die(f"job {job['job_id']}: {row.count(ids['image'])} image tokens in the prompt, expected "
            f"{IMAGE_TOKENS_PER_IMAGE} x {n_images} images: image not ingested")
    pv = enc.get("pixel_values")
    if (0 if pv is None else int(pv.shape[0])) != n_images:
        die(f"job {job['job_id']}: {0 if pv is None else int(pv.shape[0])} pixel arrays for {n_images} images")
    enc["_n_prompt_tokens"] = len(row)
    enc["_prompt_ids_sha256"] = ids_sha256(row)
    return enc


def token_type_ids_source(processor):
    """Whether this transformers version's processor returns token_type_ids itself (recorded in the run signature)."""
    from PIL import Image
    enc = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image", "image": Image.new("RGB", (224, 224))}, {"type": "text", "text": "x"}]}],
        add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    return ("processor" if "token_type_ids" in enc else "built from the image-token positions (the processor returned none)") \
        + "; checked to mark exactly the image tokens"


def collate(encs, pad_id):
    """Left-pad a list of single-job encodings into one batch (generation needs the prompt flush right)."""
    import torch
    width = max(e["input_ids"].shape[1] for e in encs)

    def lpad(x, value):
        return torch.nn.functional.pad(x, (width - x.shape[1], 0), value=value)

    batch = {"input_ids": torch.cat([lpad(e["input_ids"], pad_id) for e in encs]),
             "attention_mask": torch.cat([lpad(e["attention_mask"], 0) for e in encs])}
    if any("token_type_ids" in e for e in encs):
        batch["token_type_ids"] = torch.cat([lpad(e["token_type_ids"], 0) if "token_type_ids" in e
                                             else torch.zeros((1, width), dtype=torch.long) for e in encs])
    pixels = [e["pixel_values"] for e in encs if e.get("pixel_values") is not None]
    if pixels:
        batch["pixel_values"] = torch.cat(pixels)
    return batch


_SAMPLER_CLASS = None


def per_job_sampler_class():
    """Built lazily so that importing this module needs neither torch nor transformers."""
    global _SAMPLER_CLASS
    if _SAMPLER_CLASS is not None:
        return _SAMPLER_CLASS
    import torch
    from transformers import LogitsProcessor, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

    class PerJobSampler(LogitsProcessor):
        """Sampling with one private random generator per batch row (= per job).

        It runs as the LAST logits processor of a greedy generate(): it picks the token itself and returns scores
        that are -inf everywhere else, so the argmax that follows selects exactly that token. The distribution is
        the one stock generate(do_sample=True) samples from: banned ids removed, then transformers' own
        Temperature / TopK / TopP warpers, applied here to one row at a time so that a row's result cannot depend
        on the shape of the batch. The draw is one float64 uniform from the job's CPU generator, mapped through
        the CDF of the kept tokens taken in ascending token-id order.
        """

        def __init__(self, seeds, temperature, top_k, top_p, banned_ids, eos_ids):
            self.generators = [torch.Generator(device="cpu").manual_seed(int(s)) for s in seeds]
            self.warpers = []
            if temperature != 1.0:
                self.warpers.append(TemperatureLogitsWarper(float(temperature)))
            if top_k:
                self.warpers.append(TopKLogitsWarper(top_k=int(top_k), min_tokens_to_keep=1))
            if top_p < 1.0:
                self.warpers.append(TopPLogitsWarper(top_p=float(top_p), min_tokens_to_keep=1))
            self.banned = [int(i) for i in banned_ids]
            self.eos = {int(i) for i in eos_ids}
            self.finished = [False] * len(seeds)
            self.n_draws = [0] * len(seeds)

        def kept(self, input_ids_row, scores_row):
            """(token ids ascending, probabilities float64 on CPU) that stock sampling would draw from."""
            row = scores_row.clone()
            if self.banned:
                row[:, self.banned] = -float("inf")
            for warper in self.warpers:
                row = warper(input_ids_row, row)
            idx = torch.nonzero(torch.isfinite(row[0]), as_tuple=False).squeeze(1)
            logits = row[0, idx].to(device="cpu", dtype=torch.float64)
            return idx.cpu(), torch.softmax(logits, dim=0)

        def __call__(self, input_ids, scores):
            if scores.shape[0] != len(self.generators):
                raise RuntimeError(f"PerJobSampler built for {len(self.generators)} rows, got {scores.shape[0]}")
            out = torch.full_like(scores, -float("inf"))
            for b in range(scores.shape[0]):
                if self.finished[b]:
                    out[b, 0] = 0.0          # generate() overwrites finished rows with the pad token anyway
                    continue
                idx, probs = self.kept(input_ids[b:b + 1], scores[b:b + 1])
                cdf = torch.cumsum(probs, dim=0)
                u = torch.rand((), generator=self.generators[b], dtype=torch.float64)
                j = int(torch.searchsorted(cdf, u * cdf[-1], right=True))
                token = int(idx[min(j, idx.numel() - 1)])
                self.n_draws[b] += 1
                out[b, token] = 0.0
                if token in self.eos:
                    self.finished[b] = True
            return out

    _SAMPLER_CLASS = PerJobSampler
    return PerJobSampler


def generate_batch(model, processor, jobs, encs, device, pixel_dtype, sampler, ids):
    """Generate for jobs that share a tier and max_new_tokens. Returns one result dict per job, in order."""
    import torch
    from transformers import LogitsProcessorList
    tier, max_new = jobs[0]["tier"], jobs[0]["max_new_tokens"]
    assert all(j["tier"] == tier and j["max_new_tokens"] == max_new for j in jobs)
    batch = collate(encs, ids["pad"])
    batch = {k: (v.to(device=device, dtype=pixel_dtype) if k == "pixel_values" else v.to(device)) for k, v in batch.items()}
    kw = {"max_new_tokens": max_new, "bad_words_ids": [[THINK_OPEN_ID]], "eos_token_id": list(ids["eos"]),
          "pad_token_id": ids["pad"], "use_cache": True, "do_sample": False}
    procs = LogitsProcessorList(list(_TEST_LOGITS_PROCESSORS))
    cfg = TIERS[tier]
    if cfg["do_sample"] and sampler == "hf":
        if len(jobs) != 1:
            die("--sampler hf shares one random stream across a batch; it is only allowed with --batch-size 1")
        torch.manual_seed(jobs[0]["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(jobs[0]["seed"])
        kw.update(do_sample=True, temperature=cfg["temperature"], top_k=cfg["top_k"], top_p=cfg["top_p"])
    elif cfg["do_sample"]:
        procs.append(per_job_sampler_class()([j["seed"] for j in jobs], cfg["temperature"], cfg["top_k"], cfg["top_p"],
                                             banned_ids=[THINK_OPEN_ID], eos_ids=ids["eos"]))
    if len(procs):
        kw["logits_processor"] = procs
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**batch, **kw)
    new = out[:, batch["input_ids"].shape[1]:].tolist()
    seconds = time.time() - t0
    results = []
    for job, row in zip(jobs, new):
        row = row[:max_new]
        cut = next((n for n, t in enumerate(row) if t in ids["eos"]), None)
        kept = row if cut is None else row[:cut + 1]
        finish = "length" if cut is None else "stop"
        if finish == "length" and len(kept) != max_new:
            die(f"job {job['job_id']}: generation ended after {len(kept)} tokens without an end token "
                f"(limit {max_new}); nothing written for this batch")
        if THINK_OPEN_ID in kept:
            die(f"job {job['job_id']}: the banned token {THINK_OPEN} was generated; thinking suppression is "
                "not working in this transformers version. Nothing written for this batch.")
        results.append({"raw_response": processor.tokenizer.decode(kept, skip_special_tokens=False),
                        "n_new_tokens": len(kept), "finish_reason": finish,
                        "seconds": round(seconds / len(jobs), 3), "batch_seconds": round(seconds, 3)})
    return results


# -------------------------------------------------------------------------------------------------- run

def make_batches(pending, batch_size):
    """Consecutive jobs with the same tier and token limit, at most batch_size per batch, order kept."""
    batches = []
    for job in pending:
        last = batches[-1] if batches else None
        if last and len(last) < batch_size and (last[0]["tier"], last[0]["max_new_tokens"]) == (job["tier"], job["max_new_tokens"]):
            last.append(job)
        else:
            batches.append([job])
    return batches


def fmt_eta(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


def run_worker(args, jobs, jobs_sha, out, prefix=""):
    """Load the model and generate every pending job of this process's shard. Returns an exit code."""
    t_start = time.time()
    versions = check_env(need_peft=bool(args.adapter_dir),
                         need_accelerate=args.device != "cpu")
    cache_path = os.path.join(os.path.dirname(out), ".infer_jobs_weight_hashes.json")
    fp = fingerprint_model(args.model_dir, cache_path, hash_weights=not args.skip_weight_hash)
    if fp["verified"] is False and not args.allow_unverified_weights:
        die(f"the weights in {args.model_dir} are not {PINNED_MODEL}@{PINNED_REVISION[:12]}: "
            f"mismatch {fp['mismatch']}, missing {fp['missing']}. Attach the pinned revision "
            "(or pass --allow-unverified-weights; the output is then marked unverified).")
    adapter = fingerprint_adapter(args.adapter_dir)
    non_final = adapter_gate(adapter, jobs, args.allow_non_final_adapter)
    trained_rev = (((adapter or {}).get("trained") or {}).get("base_model") or {}).get("revision")
    if trained_rev and trained_rev != args.model_revision and not args.allow_unverified_weights:
        die(f"the adapter was trained on revision {trained_rev}, this run declares {args.model_revision}")
    gpus = gpu_info()
    precision = resolve_precision(args, adapter, gpus)
    if precision["mode"] == "4bit":
        if versions.get("bitsandbytes") is None and precision["as_trained"] is None:
            die("4-bit loading was chosen (" + precision["reason"] + ") but bitsandbytes is not installed. Install it "
                "from a private wheels dataset (kaggle_push.py --pip-install), or choose --precision bf16 (emulated on a "
                "T4) or --precision fp32 --device split (float32 over both GPUs, one worker).")
        versions = check_env(need_peft=bool(args.adapter_dir), need_bnb=True, need_accelerate=True)
    if precision["as_trained"] is False:
        log(f"WARNING: the base model is loaded as {precision['mode']} / {precision['compute_dtype']}, but the adapter was "
            f"trained with {adapter['trained']['precision']}. The evaluation should load it the same way (--precision auto).", prefix)
    log(f"precision: {precision['weights_dtype']}, arithmetic {precision['compute_dtype']} ({precision['reason']})", prefix)
    processor = load_processor(args.model_dir, args.image_backend)
    image_processor = type(processor.image_processor).__name__
    trained_ip = ((adapter or {}).get("trained") or {}).get("image_processor_class")
    if trained_ip and trained_ip != image_processor:
        log(f"WARNING: training used the image processor {trained_ip}, this run uses {image_processor}; "
            "pixel values can differ in the last bit (see --image-backend)", prefix)
    # Everything that can change a result goes into `run`, and its hash is the run signature. Paths, device
    # strings and GPU names stay out of it (they are logged per line in run_static).
    run = {"model": PINNED_MODEL, "model_revision": args.model_revision, "weights_verified": fp["verified"],
           "weights_sha256": {n: fp["sha256"].get(n) for n in WEIGHT_FILES},
           "config_git_blob": fp["git_blob"].get("config.json"),
           "chat_template_git_blob": fp["git_blob"].get("chat_template.jinja"),
           "tokenizer_sha256": fp["sha256"].get("tokenizer.json"),
           "adapter_sha256": adapter["sha256"] if adapter else None,
           "adapter_config_sha256": adapter["config_sha256"] if adapter else None,
           "adapter_is_final_protocol_adapter": (not non_final) if adapter else None,
           "allow_non_final_adapter": bool(args.allow_non_final_adapter),
           "images_verified": not args.no_verify_images,
           "token_type_ids_source": token_type_ids_source(processor),
           "precision": {k: precision[k] for k in ("mode", "weights_dtype", "compute_dtype", "quantization",
                                                   "quantized_scope", "vision_tower")},
           "attn_implementation": args.attn_implementation, "image_processor": image_processor,
           "batch_size": args.batch_size, "sampler": args.sampler, "decoding": TIERS,
           "thinking_suppressed_token_id": THINK_OPEN_ID, "versions": versions}
    signature = hashlib.sha256(json.dumps(run, sort_keys=True).encode()).hexdigest()[:16]

    mine = [j for n, j in enumerate(jobs) if args.shard is None or n % args.shard[1] == args.shard[0]]
    write_path = out if args.shard is None else f"{out}.shard{args.shard[0]}of{args.shard[1]}"
    os.makedirs(os.path.dirname(write_path) or ".", exist_ok=True)
    known = {j["job_id"] for j in jobs}

    def usable(rec, path, strict):
        """Is this earlier answer one of ours, from this job file and these settings? strict: stop if not."""
        problem = None
        if rec["job_id"] not in known:
            problem = f"holds job {rec['job_id']!r}, which is not in {args.jobs}: it belongs to another job file"
        elif rec.get("jobs_sha256") != jobs_sha:
            problem = "was written from a different version of the job file (jobs_sha256 differs)"
        elif rec.get("run_signature") != signature and not args.allow_mixed_runs:
            problem = ("was started with different settings (model, adapter, precision, batch size, sampler, image "
                       f"verification or library versions): {json.dumps(rec.get('run'), sort_keys=True)[:500]} ...")
        if problem and strict:
            die(f"{path} {problem}. Use a new --tag or --out"
                + (", or pass --allow-mixed-runs to continue anyway (import_results.py refuses a mixed file by default)."
                   if "settings" in problem else "."))
        return problem is None

    done = {}
    for path in [out] + shard_files(out):
        for rec in read_output(path, repair=(path == write_path)):
            usable(rec, path, strict=True)
            done[rec["job_id"]] = rec
    carried, skipped, mine_ids = [], 0, {j["job_id"] for j in mine}
    for path in earlier_answers(args.resume_from, out):          # an earlier session's output, read-only
        for rec in read_output(path):
            if rec["job_id"] in done or rec["job_id"] not in mine_ids:
                continue
            if usable(rec, path, strict=False):
                done[rec["job_id"]] = rec
                carried.append(rec)
            else:
                skipped += 1
    if carried:
        order = {j["job_id"]: n for n, j in enumerate(jobs)}
        with open(write_path, "a") as fh:
            for rec in sorted(carried, key=lambda r: order[r["job_id"]]):
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    if args.resume_from:
        log(f"--resume-from: {len(carried)} finished answers carried over"
            + (f", {skipped} ignored (other job file version or other settings)" if skipped else ""), prefix)
    pending = [j for j in mine if j["job_id"] not in done]
    log(f"{len(mine)} jobs in this shard, {len(mine) - len(pending)} already done, {len(pending)} to run -> {write_path}", prefix)
    if not pending:
        return 0

    ids = special_ids(processor)
    model, device, dtype = load_model(args, versions, precision)
    run_static = {"model_class": type(model).__name__, "gpus": gpus["gpus"], "device": str(device),
                  "device_arg": args.device, "model_dir": args.model_dir, "adapter": adapter,
                  "precision_reason": precision["reason"], "precision_as_trained": precision["as_trained"]}
    n_done, n_tokens, gen_seconds, t_loop = 0, 0, 0.0, time.time()
    with open(write_path, "a") as fh:
        for batch_jobs in make_batches(pending, args.batch_size):
            if args.time_budget_hours and (time.time() - t_start) / 3600 >= args.time_budget_hours:
                log(f"time budget of {args.time_budget_hours:g} h reached with {len(pending) - n_done} jobs left", prefix)
                return EXIT_TIME_BUDGET
            encs = [encode_job(processor, j, args.bundle_dir, ids, verify_images=not args.no_verify_images) for j in batch_jobs]
            try:
                results = generate_batch(model, processor, batch_jobs, encs, device, dtype, args.sampler, ids)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    die(f"GPU out of memory at batch size {args.batch_size} ({max(e_['_n_prompt_tokens'] for e_ in encs)} "
                        "prompt tokens). Re-run with a smaller --batch-size and a new --tag.", code=4)
                raise
            for job, enc, res in zip(batch_jobs, encs, results):
                rec = {"job_id": job["job_id"], "raw_response": res["raw_response"],
                       "n_prompt_tokens": enc["_n_prompt_tokens"], "n_new_tokens": res["n_new_tokens"],
                       "finish_reason": res["finish_reason"], "seconds": res["seconds"],
                       "tier": job["tier"], "seed": job["seed"], "max_new_tokens": job["max_new_tokens"],
                       "n_images": n_images_of(job), "prompt_ids_sha256": enc["_prompt_ids_sha256"],
                       "model_revision": run["model_revision"], "weights_verified": run["weights_verified"],
                       "images_verified": run["images_verified"], "adapter_sha256": run["adapter_sha256"],
                       "dtype": precision["compute_dtype"], "quantization": precision["quantization"],
                       "batch_size": args.batch_size, "batch_rows": len(batch_jobs), "batch_seconds": res["batch_seconds"],
                       "sampler": args.sampler if TIERS[job["tier"]]["do_sample"] else "greedy",
                       "run_signature": signature, "jobs_sha256": jobs_sha, "run": {**run, **run_static},
                       "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                fh.write(json.dumps(rec) + "\n")
                n_done += 1
                n_tokens += res["n_new_tokens"]
            fh.flush()
            os.fsync(fh.fileno())
            gen_seconds += results[0]["batch_seconds"]
            per_job = (time.time() - t_loop) / n_done
            last = results[-1]
            log(f"[{len(mine) - len(pending) + n_done}/{len(mine)}] {batch_jobs[-1]['job_id']} {last['finish_reason']} "
                f"{last['n_new_tokens']} tok | {per_job:.1f} s/job, {n_tokens / max(gen_seconds, 1e-9):.1f} tok/s, "
                f"{3600 / per_job:.0f} jobs/h | ETA {fmt_eta(per_job * (len(pending) - n_done))}", prefix)
    log(f"shard finished: {n_done} new jobs, {n_tokens} new tokens, {time.time() - t_loop:.0f}s", prefix)
    return 0


def script_path():
    """This script as a file a worker process can run. Inside a notebook kernel there is no file: the source of
    the running cell is written to a temporary one. None if neither works (then one worker is used)."""
    path = globals().get("__file__")
    if path and os.path.isfile(path):
        return os.path.abspath(path)
    try:
        source = get_ipython().user_ns["_ih"][-1]                  # noqa: F821 - only defined inside IPython
        if "def run_workers(" not in source or "def per_job_sampler_class(" not in source:
            return None
        path = os.path.join(tempfile.gettempdir(), "infer_jobs_worker.py")
        with open(path, "w") as fh:
            fh.write(source)
        return path
    except Exception:   # noqa: BLE001 - no IPython, or no cell history
        return None


def child_argv(args, script, out, i, n):
    argv = [sys.executable, script, "--jobs", args.jobs, "--model-dir", args.model_dir,
            "--bundle-dir", args.bundle_dir, "--out", out, "--model-revision", args.model_revision,
            "--precision", args.precision, "--compute-dtype", args.compute_dtype,
            "--attn-implementation", args.attn_implementation, "--image-backend", args.image_backend,
            "--batch-size", str(args.batch_size), "--sampler", args.sampler,
            "--device", "cpu" if args.device == "cpu" else "cuda", "--workers", "1", "--shard", f"{i}/{n}",
            "--time-budget-hours", str(args.time_budget_hours), "--exit-zero", "no"]
    if args.adapter_dir:
        argv += ["--adapter-dir", args.adapter_dir]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    for src in args.resume_from or []:
        argv += ["--resume-from", src]
    for flag, on in (("--skip-weight-hash", args.skip_weight_hash),
                     ("--allow-unverified-weights", args.allow_unverified_weights),
                     ("--allow-mixed-runs", args.allow_mixed_runs), ("--no-verify-images", args.no_verify_images),
                     ("--allow-non-final-adapter", args.allow_non_final_adapter)):
        if on:
            argv.append(flag)
    return argv


def run_workers(args, jobs, out, n, script):
    """One process per GPU. Each writes its own shard file; the shards are merged when all have exited."""
    cache_path = os.path.join(os.path.dirname(out), ".infer_jobs_weight_hashes.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fingerprint_model(args.model_dir, cache_path, hash_weights=not args.skip_weight_hash)   # hash once, children reuse it
    procs = []
    for i in range(n):
        env = dict(os.environ)
        if args.device != "cpu":
            env["CUDA_VISIBLE_DEVICES"] = str(i)
        procs.append(subprocess.Popen(child_argv(args, script, out, i, n), env=env))
        log(f"worker {i}/{n} started (pid {procs[-1].pid})")
    codes = [p.wait() for p in procs]
    merge_shards(out, [j["job_id"] for j in jobs])
    return max(codes) if all(c >= 0 for c in codes) else 1


def compare(path_a, path_b):
    """Share of jobs with identical raw_response in two output files (e.g. batch size 1 vs 4)."""
    a = {r["job_id"]: r for r in read_output(path_a)}
    b = {r["job_id"]: r for r in read_output(path_b)}
    common = sorted(set(a) & set(b))
    if not common:
        die("the two files share no job_id")
    same = [j for j in common if a[j]["raw_response"] == b[j]["raw_response"]]
    log(f"{len(common)} jobs in both files ({len(a)} / {len(b)}); identical raw_response: {len(same)} "
        f"({100 * len(same) / len(common):.1f}%)")
    for j in common:
        if a[j]["raw_response"] != b[j]["raw_response"]:
            x, y = a[j]["raw_response"], b[j]["raw_response"]
            k = next((n for n, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            log(f"  differs: {j} at character {k}: {x[k:k + 40]!r} vs {y[k:k + 40]!r}")
    return 0 if len(same) == len(common) else 1


def finalize(out, jobs, code, message=None):
    """Print the summary, write <out>.status.json, and return (status, number of answers on disk)."""
    recs = {r["job_id"]: r for r in read_output(out)}
    got = [recs[j["job_id"]] for j in jobs if j["job_id"] in recs]
    if code == 0 and len(got) == len(jobs):
        status = "complete"
    else:
        status = "incomplete" if code in (0, EXIT_TIME_BUDGET) else "failed"
    if got:
        stops = sum(r["finish_reason"] == "stop" for r in got)
        log(f"{out}: {len(got)}/{len(jobs)} jobs, finish stop {stops} / length {len(got) - stops}, "
            f"mean {sum(r['n_new_tokens'] for r in got) / len(got):.0f} new tokens, "
            f"{sum(r['seconds'] for r in got) / len(got):.1f} s per job")
    try:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(re.sub(r"\.jsonl$", "", out) + ".status.json", "w") as fh:
            json.dump({"status": status, "n_done": len(got), "n_jobs": len(jobs), "exit_code": code, "message": message,
                       "output": os.path.basename(out), "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, fh, indent=1)
    except OSError:
        pass
    return status, len(got)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", help="job file (.jsonl) built by build_dev_jobs.py, e.g. /kaggle/input/<jobs>/<name>.jsonl")
    ap.add_argument("--model-dir", default=os.environ.get("MEDGEMMA_DIR"),
                    help="folder with the official medgemma-1.5-4b-it files (config.json, the two safetensors, "
                         "tokenizer/processor files). Env: MEDGEMMA_DIR")
    ap.add_argument("--adapter-dir", default=os.environ.get("LORA_ADAPTER_DIR") or None,
                    help="optional PEFT LoRA adapter folder (adapter_config.json + adapter_model.safetensors, e.g. "
                         "train_lora.py's best_adapter/); loaded on top of the base weights, NOT merged. Env: LORA_ADAPTER_DIR")
    ap.add_argument("--bundle-dir", default=os.environ.get("MHIST_BUNDLE_DIR"),
                    help="data bundle: the folder the image paths in the job file are relative to, i.e. the one "
                         "holding gridded/ (default: found next to the job file or among the attached datasets). "
                         "Env: MHIST_BUNDLE_DIR")
    ap.add_argument("--out", help="output .jsonl (default: /kaggle/working/<job file stem>__<tag>.jsonl, or the "
                                  "current folder outside Kaggle)")
    ap.add_argument("--tag", help="suffix of the default output name (default: 'base', or 'lora-<first 8 hex of the adapter hash>')")
    ap.add_argument("--resume-from", action="append", metavar="PATH",
                    help="answer file of an earlier session, or a folder searched for files named like --out "
                         "(e.g. the attached output of the previous notebook); finished answers are copied first. Repeatable")
    ap.add_argument("--model-revision", default=PINNED_REVISION,
                    help="revision string written to the output; the weights are checked against the pinned "
                         "revision's published hashes either way (default: %(default)s)")
    ap.add_argument("--precision", default="auto", choices=["auto", "bf16", "4bit", "fp32"],
                    help="auto = as the adapter was trained (step4_meta.json); without that, bf16 on a GPU with native "
                         "bf16, else 4-bit nf4 (bitsandbytes). Recorded in every line")
    ap.add_argument("--compute-dtype", default="auto", choices=["auto", "bf16", "fp32"],
                    help="4-bit only: arithmetic dtype. auto = as trained, else bf16 on native-bf16 GPUs and fp32 on a T4")
    ap.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa"],
                    help="attention implementation (default eager, as train_lora.py uses)")
    ap.add_argument("--image-backend", default="default", choices=["default", "pil"],
                    help="default = AutoProcessor's own image processor (as train_lora.py); pil = force the PIL pipeline")
    ap.add_argument("--device", default="cuda",
                    help="cuda (default; one GPU per worker), cuda:N, cpu, or split (one process over all GPUs, "
                         "needed for fp32 on 16 GB cards)")
    ap.add_argument("--workers", default="auto",
                    help="processes, one per GPU (default auto: the number of visible GPUs when --device is cuda, else 1)")
    ap.add_argument("--shard", help="i/n: only jobs whose index %% n == i, written to <out>.shard<i>of<n> (set by --workers)")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="jobs per generate() call (default 1; see RESIDUAL BATCH DEPENDENCE above)")
    ap.add_argument("--sampler", default="per-job", choices=["per-job", "hf"],
                    help="tier-1 sampler: per-job private generators (default), or stock HF sampling after "
                         "torch.manual_seed(seed) (batch size 1 only)")
    ap.add_argument("--limit", type=int, help="only the first N jobs of the file (timing run)")
    ap.add_argument("--time-budget-hours", type=float, default=11.0,
                    help="stop cleanly once this many hours have passed in this session (Kaggle limit: 12); "
                         "resume with --resume-from in the next one (default %(default)s)")
    ap.add_argument("--exit-zero", default="auto", choices=["auto", "yes", "no"],
                    help="Kaggle discards /kaggle/working when a notebook exits non-zero. With yes, a run that stops "
                         "early but has written answers exits 0 (the status file and the log say it is incomplete). "
                         "auto = yes when /kaggle/working exists")
    ap.add_argument("--skip-weight-hash", action="store_true",
                    help="do not sha256 the two weight files (about a minute); weights_verified is then null and "
                         "import_results.py needs --allow-unverified-weights")
    ap.add_argument("--allow-unverified-weights", action="store_true",
                    help="run even if the weights do not match the pinned revision (output marked weights_verified=false)")
    ap.add_argument("--allow-mixed-runs", action="store_true",
                    help="resume an output file that was started with different settings")
    ap.add_argument("--no-verify-images", action="store_true",
                    help="skip the sha256 check of bundle images; recorded as images_verified=false in every output "
                         "line, and import_results.py then needs --allow-unverified-images")
    ap.add_argument("--allow-non-final-adapter", action="store_true",
                    help="run dev jobs with an adapter that is not the final best_adapter/ of a protocol run (an epoch "
                         "checkpoint, an unfinished or debug run, no step4_meta.json). Recorded in every output line; "
                         "import_results.py refuses such answers. Not for the protocol's dev evaluation")
    ap.add_argument("--dry-run", action="store_true",
                    help="no model: tokenize every job with the processor, check images and image-token counts, print stats")
    ap.add_argument("--print-env", action="store_true", help="print library versions and GPUs, then exit")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare raw_response between two output files, then exit")
    if argv is None:
        argv = sys.argv[1:]
        if argv[:1] == ["-f"] or (not argv and os.environ.get("INFER_JOBS_ARGS")):
            argv = shlex.split(os.environ.get("INFER_JOBS_ARGS", ""))   # started by a notebook kernel without our arguments
    args = ap.parse_args(argv)
    if args.compare or args.print_env:
        return args
    if not args.jobs:
        ap.error("--jobs is required")
    if args.batch_size < 1:
        ap.error("--batch-size must be >= 1")
    if args.sampler == "hf" and args.batch_size != 1:
        ap.error("--sampler hf is only allowed with --batch-size 1")
    if args.shard:
        try:
            i, n = (int(x) for x in args.shard.split("/"))
            assert 0 <= i < n
        except (ValueError, AssertionError):
            ap.error("--shard must be i/n with 0 <= i < n")
        args.shard = (i, n)
    return args


def main(argv=None):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")          # never reach for the network
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if args.print_env:
        log(json.dumps({"versions": check_env(), "gpu": gpu_info()}, indent=1))
        return 0
    args.jobs = os.path.abspath(args.jobs)
    jobs, jobs_sha = read_jobs(args.jobs)
    if args.limit:
        jobs = jobs[:args.limit]
    args.model_dir = find_dir_with(args.model_dir, "config.json", "--model-dir / MEDGEMMA_DIR")
    if args.adapter_dir:
        args.adapter_dir = find_dir_with(args.adapter_dir, "adapter_config.json", "--adapter-dir / LORA_ADAPTER_DIR")
    if args.bundle_dir:
        args.bundle_dir = os.path.abspath(args.bundle_dir)
    elif any(n_images_of(j) for j in jobs):
        args.bundle_dir = default_bundle_dir(args.jobs)
    else:
        args.bundle_dir = os.path.dirname(args.jobs)          # text-only jobs need no bundle

    if args.adapter_dir:                             # before anything is read or loaded: the right adapter for dev jobs?
        adapter_gate(fingerprint_adapter(args.adapter_dir), jobs, args.allow_non_final_adapter)

    if args.dry_run:
        check_env()
        processor = load_processor(args.model_dir, args.image_backend)
        ids = special_ids(processor)
        lengths = [encode_job(processor, j, args.bundle_dir, ids, verify_images=not args.no_verify_images)["_n_prompt_tokens"]
                   for j in jobs]
        log(f"dry run OK: {len(jobs)} jobs, prompt tokens {min(lengths)}..{max(lengths)}, images per job "
            f"{sorted({n_images_of(j) for j in jobs})}, tiers {sorted({j['tier'] for j in jobs})}, "
            f"image processor {type(processor.image_processor).__name__}")
        return 0

    if not args.out:
        tag = args.tag or ("lora-" + fingerprint_adapter(args.adapter_dir)["sha256"][:8] if args.adapter_dir else "base")
        base = KAGGLE_WORKING if os.path.isdir(KAGGLE_WORKING) else os.getcwd()
        args.out = os.path.join(base, f"{os.path.splitext(os.path.basename(args.jobs))[0]}__{re.sub(r'[^A-Za-z0-9._-]+', '-', tag)}.jsonl")
    out = os.path.abspath(args.out)

    if args.shard is not None:                       # a worker started by run_workers (or by hand): plain exit code
        return run_worker(args, jobs, jobs_sha, out, prefix=f"[w{args.shard[0]}] ")

    if args.workers == "auto":
        import torch
        n = torch.cuda.device_count() if (args.device == "cuda" and torch.cuda.is_available()) else 1
    else:
        n = int(args.workers)
        if n > 1 and args.device not in ("cuda", "cpu"):
            die("--workers > 1 gives each worker one whole GPU; use it with --device cuda (or cpu), not with "
                f"--device {args.device}")
    n = max(1, min(n, len(jobs)))
    script = script_path() if n > 1 else None
    if n > 1 and script is None:
        log("note: this kernel has no script file to start worker processes from; running one worker on one GPU")
        n = 1
    merge_shards(out, [j["job_id"] for j in jobs])   # fold in shard files left by an interrupted run
    log(f"{len(jobs)} jobs from {args.jobs} (sha256 {jobs_sha[:12]}), {n} worker(s), batch size {args.batch_size}, "
        f"sampler {args.sampler}, precision {args.precision} -> {out}")
    code, message = 0, None
    try:
        code = run_worker(args, jobs, jobs_sha, out) if n == 1 else run_workers(args, jobs, out, n, script)
    except KeyboardInterrupt:
        raise
    except SystemExit as e:                          # die(): the message is already on stderr
        code, message = (e.code if isinstance(e.code, int) else 2), "stopped by a check; see the log"
    except Exception as e:                           # noqa: BLE001 - keep what was written, report, decide the exit code
        traceback.print_exc()
        code, message = 1, f"{type(e).__name__}: {str(e)[:300]}"
        with_shards = shard_files(out)
        if with_shards:
            merge_shards(out, [j["job_id"] for j in jobs])
    status, n_done = finalize(out, jobs, code, message)
    if status == "complete":
        return 0
    keep = args.exit_zero == "yes" or (args.exit_zero == "auto" and os.path.isdir(KAGGLE_WORKING))
    log(f"NOT FINISHED: {status}, {n_done}/{len(jobs)} answers on disk (real exit code {code or 1})."
        + (" Start a new session with this output attached and --resume-from <its folder>, or re-run here." if n_done else ""))
    if keep and n_done:
        log("Exit status 0 on purpose, so that Kaggle keeps the answers written so far; see the status file.")
        return 0
    return code or 1


if __name__ == "__main__":
    _exit_code = main()
    if _exit_code:                                   # a plain return on success: inside a notebook kernel sys.exit(0) is reported as an error
        sys.exit(_exit_code)
