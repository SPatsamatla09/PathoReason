#!/usr/bin/env python3
"""Step-4 LoRA fine-tune of MedGemma 1.5 4B on the MHIST training pool (runs on a PRIVATE Kaggle GPU notebook).

This is the Kaggle-side trainer for "Addendum: step 4" of runs/competence/PLAN.md. Everything the addendum
fixes is a constant in this file, not an argument, so a run cannot drift from the protocol by accident:

    base model      google/medgemma-1.5-4b-it, revision 91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b (attached, never downloaded)
    training set    fewshot_pool (1,875 tiles = train split minus dev); the frozen name list is hash-checked
    validation      15% of the pool, stratified by label x agreement band, seed 20261004 -> <out-dir>/split.json
    input           gridded 224x224 tile + the pipeline's cte_p1 prompt, official chat format (image first, no system prompt)
    target          '{\\n  "label": "HP"' or '{\\n  "label": "SSA"'; loss on those target tokens only
                    (TARGET_VARIANT below; the alternative that stops at the label needs a PLAN amendment first)
    imbalance       class-weighted loss, weight = N_train / (2 * n_class) (inverse frequency, mean weight 1)
    LoRA            r=16, alpha=16, dropout 0.05 on the language model's linear layers only; vision tower frozen
    optimiser       AdamW, lr 2e-4, cosine schedule with 5% linear warm-up, effective batch 8 by accumulation
    epochs          4; after each epoch the validation balanced accuracy is read from the HP-vs-SSA logits at the
                    label position; the adapter with the best value is kept (ties: the earlier epoch)
    seed            20261004

Example (private Kaggle script notebook, started by kaggle_push.py; internet may be off):

    python3 kaggle_ft/kaggle_push.py kernel-push --script kaggle_ft/train_lora.py --slug <slug> \\
        --dataset bundle --dataset weights \\
        --args='--model-dir {weights} --data-dir {bundle} --out-dir /kaggle/working/step4_lora'

    or, in a notebook cell:  !python train_lora.py --model-dir /kaggle/input/<weights> --data-dir /kaggle/input/<bundle>
    (without --model-dir / --data-dir the one matching dataset under /kaggle/input is used)
    A notebook kernel that starts this file without arguments (argv is `-f <connection file>`) takes them from the
    environment variable TRAIN_LORA_ARGS. kaggle_push.py does not rely on that: its header sets sys.argv itself.
    The first log line shows the arguments that were actually parsed: read it before trusting a run.

    --verify-only   checks the bundle, the split, the packages and the tokenisation, prints the plan, loads no weights
    re-running the same command resumes from <out-dir>/checkpoints/last (saved every few optimiser steps and at
    every epoch end); a finished run is detected and skipped. To resume in a NEW Kaggle session, attach the
    earlier kernel's output as an input: --resume-from auto (the default) finds the checkpoint under
    /kaggle/input and copies it into --out-dir. The script stops by itself after --time-budget-hours (11) with a
    checkpoint. On Kaggle a pause or a failure with a checkpoint exits 0, because a non-zero exit discards
    /kaggle/working; read train_summary.json 'status' (complete / paused / failed) and FAILED.txt.

Inputs (never written to):
    <model-dir>/            config.json, model-0000x-of-00002.safetensors, tokenizer/processor files
    <data-dir>/gridded/     one gridded PNG per POOL tile (rendered on the Mac by run_experiment.b64_gridded_tile)
    <data-dir>/train_manifest.csv   columns image,label,ssa_votes -- POOL tiles only
    <data-dir>/prompts/cte_p1.txt   the pipeline prompt (sha256 is checked)

Outputs (all under <out-dir>):
    split.json              train/validation tile names
    train_log.jsonl         one line per epoch (plus an untrained baseline line, epoch 0)
    train_steps.jsonl       one line every few optimiser steps (loss, lr, speed, ETA)
    val_scores/epoch_N.json per-tile HP/SSA log-probabilities at the label position
    checkpoints/epoch_N/    the adapter after each epoch (PEFT format)
    checkpoints/last/       resume state (trainable weights, optimiser, counters)
    best_adapter/           copy of the best epoch's adapter + step4_meta.json. While the run is going this is the
                            best SO FAR ("final": false). Only when every epoch has been validated is it marked
                            "final": true with chosen_epoch, epochs_completed and adapter_model_sha256, and only
                            such an adapter of a protocol run is accepted for the dev evaluation (infer_jobs.py,
                            import_results.py). checkpoints/epoch_N/ adapters are never final.
    train_summary.json      status (complete only after best_adapter is final), chosen epoch, per-epoch validation
                            metrics, class weights, dtype/quantisation, package versions, wall time
    FAILED.txt              only after an error: message and traceback

No dev or test tile is read: the manifest must hash to the frozen pool list, and only tiles named in the
manifest are opened (the shared bundle also holds the dev tiles for the later evaluation; they are left alone).
Nothing is uploaded, pushed or downloaded by this script (HF_HUB_OFFLINE=1 is set before any import).

Hardware: --precision auto loads bf16 weights when the GPU has native bf16 (compute capability >= 8, the test
in Google's notebook). Kaggle's T4 does not, so there the base model is loaded 4-bit nf4 (bitsandbytes, double
quantisation) with float32 arithmetic (fp16 overflows in Gemma 3; bf16 is only emulated on a T4). Which one was
used is in train_summary.json ('precision', 'used_4bit') and in best_adapter/step4_meta.json: the evaluation must
load the base model the same way.
4-bit scope: only what memory requires (the addendum's condition). The language model's linear layers are
quantised (float32 they would need 12.8 GB on a 15 GB T4); the frozen SigLIP vision tower (1.7 GB float32), the
projector, the embeddings and lm_head are NOT. This is checked on the loaded model (exactly 7 x 34 Linear4bit
modules, all inside the language model) and recorded as precision.quantized_scope / precision.vision_tower.

Two implementation choices that do not change the mathematics:
    * the vision tower and projector are frozen, so their output for a tile (256 x 2560) is computed once and
      reused in every epoch; before that, the cached route is checked against the ordinary pixel route on this
      model and dropped if the logits differ (--feature-cache auto|on|off);
    * only the logits at the target positions are computed (logits_to_keep), not all 831 x 262,208.
Order of examples, dropout masks and the split are pure functions of the seed (and the step), so a resumed run
repeats exactly what an uninterrupted run would have done.

Packages (checked at start, with a clear message): torch >= 2.2, transformers >= 4.50 (Gemma 3; the model was
saved with 4.57.1), peft >= 0.13, safetensors, accelerate >= 0.26 on GPU, and bitsandbytes >= 0.43 on a GPU
without native bf16.

Conventions taken from Google's official notebook
(github.com/google-health/medgemma, notebooks/fine_tune_with_hugging_face.ipynb, read 2026-10-04):
    * AutoModelForImageTextToText.from_pretrained(..., attn_implementation="eager", torch_dtype=bfloat16, device_map=...)
    * the bf16 capability test `torch.cuda.get_device_capability()[0] < 8`
    * BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=...) for the 4-bit path (plus llm_int8_skip_modules here, see "4-bit scope" above)
    * LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    * collate: text from processor.apply_chat_template(..., tokenize=False), images passed as [[img.convert("RGB")]],
      then processor(text=..., images=..., return_tensors="pt"); labels = -100 on everything not trained on
    * training: lr 2e-4, gradient checkpointing with use_reentrant=False, fused AdamW, max_grad_norm 0.3,
      bf16 autocast, right padding
Deliberate differences from that notebook (each forced by the PLAN addendum or by exactness):
    * target_modules is NOT "all-linear" (that would also wrap the SigLIP vision tower): an explicit list of the
      language model's nn.Linear layers is built from the module tree and verified after wrapping
    * no modules_to_save=["lm_head","embed_tokens"] (the addendum trains LoRA matrices only)
    * add_generation_prompt=True and the target appended as raw tokens; add_special_tokens=False, because the
      notebook's collate tokenises the templated text with special tokens and so yields a second <bos>
      (checked with transformers 5.18); here the prompt is token-identical to apply_chat_template(tokenize=True)
    * loss on the target tokens only (the notebook also trains on the prompt text), class-weighted
    * cosine schedule with 5% warm-up, effective batch 8, 4 epochs (notebook: linear, 3%, 16, 1)
    * bnb_4bit_quant_storage is left at its default (the notebook's setting only matters for FSDP sharding)
    * a plain PyTorch loop instead of trl.SFTTrainer (one package fewer to need with internet off, and the
      per-example class weights and the label read-out need a custom loss and evaluation anyway)
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import inspect
import json
import math
import os
import shlex
import shutil
import sys
import tempfile
import time
import traceback
from fractions import Fraction

# Offline and quiet before transformers / torch are imported (they are imported lazily, inside functions).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS

# --------------------------------------------------------------------------------------------------------------
# Frozen protocol constants (PLAN.md, "Addendum: step 4"). Not arguments on purpose.
# --------------------------------------------------------------------------------------------------------------
SEED = 20261004
VAL_FRACTION = Fraction(15, 100)
CLASSES = ("HP", "SSA")
# The supervised assistant text. "plan" is the addendum as written: 9 tokens, the last one the bare quote '"'.
# In a full cte_p1 answer the label is followed by the single token '",' instead, so "plan" supervises a token
# the full answer never uses; whether the adapter still writes confidence and evidence after it is what the
# label-free smoke check (README, "Format check before dev") measures before any dev tile is touched.
# "label_end" stops at the label (8 tokens; nothing after the label is supervised, the model's own '",' is kept).
# It changes the PLAN's target text: switch to it ONLY after a dated PLAN amendment, and before any training.
TARGET_TEMPLATES = {"plan": '{{\n  "label": "{label}"', "label_end": '{{\n  "label": "{label}'}
TARGET_VARIANT = "label_end"
TARGET_TEMPLATE = TARGET_TEMPLATES[TARGET_VARIANT]
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 16, 0.05
LEARNING_RATE = 2e-4
WARMUP_FRACTION = 0.05
EFFECTIVE_BATCH = 8
MAX_EPOCHS = 4
# Not fixed by the addendum; taken from Google's notebook / HF defaults and recorded in train_summary.json.
MAX_GRAD_NORM = 0.3
ADAM_BETAS, ADAM_EPS, WEIGHT_DECAY = (0.9, 0.999), 1e-8, 0.0

MODEL_ID = "google/medgemma-1.5-4b-it"
MODEL_REVISION = "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"
# sha256 of the small files of that revision and byte sizes of the weight shards (from the local official copy).
EXPECTED_MODEL_FILES = {
    "config.json": "300c724c2c1fcdea39f1e21865cb1b14a605f1e3bb5ef50550faaa48da944fc8",
    "model.safetensors.index.json": "77f4b67de084c31c7bcd373b039908108eee6c6181607e6d53da730e5f0bc659",
    "tokenizer.json": "7d4046bf0505a327dd5a0abbb427ecd4fc82f99c2ceaa170bc61ecde12809b0c",
    "chat_template.jinja": "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4",
    "preprocessor_config.json": "5b2f684b616a25f3cd4a700e5e471a07fcaabc5ec07871d2231b7e376e8648ce",
}
EXPECTED_MODEL_SHARD_BYTES = {
    "model-00001-of-00002.safetensors": 4961251752,
    "model-00002-of-00002.safetensors": 3639026128,
}
EXPECTED_POOL_N = 1875
# sha256 of "\n".join(sorted(fewshot_pool)) from runs/competence/splits.json: image names only, no labels.
EXPECTED_POOL_SHA256 = "20c9cbd5a0e4bb79c30354cf5e2f3dbb5c1dfdc401e4f5dd31680e0bf0c6c402"
# sha256 of "\n".join(validation tile names) that make_split gives for the frozen pool (checked on the Mac with
# Python 3.9 and 3.13): a Kaggle run must reproduce exactly this validation slice.
EXPECTED_VAL_SHA256 = "156dd1f1c097f99701d32fa650ff103d45312e34fffc91a62ee7e0c0b926e2a5"
# sha256 of prompts/rendered/cte_p1.txt (comp_run.py logs its first 16 hex digits as prompt_sha256).
EXPECTED_PROMPT_SHA256 = "eb0e4d85b580cebad7c1733b4d5e80bee6b28c322eb48032da169b7a7917654f"
EXPECTED_PROMPT_TOKENS = 822           # cte_p1 + one image under the official template (PLAN, amendment 2)
EXPECTED_IMAGE_TOKENS = 256
TILE_SIZE = (224, 224)

LM_LINEAR_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# Modules bitsandbytes must leave alone on the 4-bit path. One name per module-tree layout and per matching rule
# of the transformers releases in use (substring match up to 4.x, regex anchored at the start of the full name
# in 5.x): with "vision_tower" alone, transformers 5.18 quantises the whole vision tower. The result is checked
# on the loaded model by check_quantized_scope(), whatever the installed version does with these names.
BNB_SKIP_MODULES = ("vision_tower", "model.vision_tower", "multi_modal_projector", "model.multi_modal_projector",
                    "lm_head", "language_model.lm_head")
QUANTIZED_SCOPE = "language-model linear layers only; vision_tower, multi_modal_projector, embeddings and lm_head not quantised"
MIN_VERSIONS = {"torch": "2.2.0", "transformers": "4.50.0", "peft": "0.13.0", "safetensors": "0.4.0",
                "accelerate": "0.26.0", "bitsandbytes": "0.43.0"}
EXIT_PAUSED = 75   # stopped on purpose with a resumable checkpoint (time budget or --debug-stop-after-steps)
KAGGLE_INPUT, KAGGLE_WORKING = "/kaggle/input", "/kaggle/working"
RUN_ITEMS = ("checkpoints", "best_adapter", "val_scores", "train_log.jsonl", "train_steps.jsonl",
             "train_summary.json", "model_files.json")   # what an earlier session's output consists of


# --------------------------------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------------------------------
def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def die(msg):
    raise SystemExit(f"ERROR: {msg}")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_sha256(obj):
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def parse_version(v):
    out = []
    for part in str(v).split("+")[0].split("."):
        digits = ""
        for ch in part:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    atomic_write_text(path, json.dumps(obj, indent=2) + "\n")


def recover_dir(final_dir):
    """Finish or undo a swap_dir that was interrupted, and drop leftovers."""
    old, tmp = final_dir + ".old", final_dir + ".tmp"
    if not os.path.isdir(final_dir) and os.path.isdir(old):
        os.rename(old, final_dir)
    for d in (old, tmp):
        if os.path.isdir(d):
            shutil.rmtree(d)


def swap_dir(tmp_dir, final_dir):
    """Replace final_dir by tmp_dir so that a complete directory exists at every instant."""
    old = final_dir + ".old"
    if os.path.isdir(old):
        shutil.rmtree(old)
    if os.path.isdir(final_dir):
        os.rename(final_dir, old)
    os.rename(tmp_dir, final_dir)
    if os.path.isdir(old):
        shutil.rmtree(old)


def append_jsonl_once(path, rec, key_fields):
    """Append rec unless a line with the same key fields exists (keeps the epoch log idempotent across resumes)."""
    key = tuple(rec.get(k) for k in key_fields)
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    old = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if tuple(old.get(k) for k in key_fields) == key:
                    return False
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return True


def resolve_dir(path, markers, what):
    """Kaggle mounts a dataset at /kaggle/input/<slug>/ and may keep a folder level or two from the upload."""
    if not path or not os.path.isdir(path):
        die(f"{what} directory not found: {path!r}. Attach the private dataset and pass its path.")

    def ok(d):
        return all(os.path.exists(os.path.join(d, m)) for m in markers)

    if ok(path):
        return path
    hits = []
    for root, dirs, _ in os.walk(path):
        if root[len(path):].count(os.sep) >= 3:
            dirs[:] = []
        if ok(root):
            hits.append(root)
    if len(hits) == 1:
        return hits[0]
    die(f"{what} directory {path!r} does not contain {markers} "
        f"({'several candidates: ' + str(hits) if hits else 'nothing found up to 3 levels down'}).")


def looks_like_model_dir(d, files):
    return "config.json" in files and "tokenizer.json" in files and any(f.endswith(".safetensors") for f in files)


def looks_like_bundle_dir(d, files):
    return "train_manifest.csv" in files and os.path.isdir(os.path.join(d, "gridded"))


def discover_dir(looks_right, flag, what):
    """No path given: take the one attached dataset under /kaggle/input that looks right."""
    hits = []
    if os.path.isdir(KAGGLE_INPUT):
        for root, dirs, files in os.walk(KAGGLE_INPUT, followlinks=True):
            if root[len(KAGGLE_INPUT):].count(os.sep) >= 4:
                dirs[:] = []
            if looks_right(root, files):
                hits.append(root)
                dirs[:] = []
    if len(hits) == 1:
        return hits[0]
    die(f"{flag} was not given and {len(hits)} attached datasets under {KAGGLE_INPUT} look like {what} "
        f"({hits[:4]}). Pass {flag}.")


def copy_tree_writable(src, dst):
    """Copy file contents only (attached Kaggle inputs are read-only; the copy must stay writable)."""
    if os.path.isfile(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        return
    for root, _, files in os.walk(src):
        target = os.path.normpath(os.path.join(dst, os.path.relpath(root, src)))
        os.makedirs(target, exist_ok=True)
        for f in files:
            shutil.copyfile(os.path.join(root, f), os.path.join(target, f))


def run_progress(state):
    return (int(state.get("global_step", 0)), len(state.get("history", [])))


def find_resumable(search_root, fingerprint, not_under=None):
    """The run directory under search_root whose checkpoints/last matches this configuration and is furthest
    along, as ((global_step, epochs_validated), run_dir), or None. An earlier Kaggle session's output, attached
    as an input, is found this way."""
    best = None
    if not search_root or not os.path.isdir(search_root):
        return None
    search_root = os.path.abspath(search_root)
    for root, dirs, files in os.walk(search_root, followlinks=True):
        if not_under and (os.path.abspath(root) + os.sep).startswith(os.path.abspath(not_under) + os.sep):
            dirs[:] = []
            continue
        if root[len(search_root):].count(os.sep) >= 6:
            dirs[:] = []
        if os.path.basename(root) == "last" and os.path.basename(os.path.dirname(root)) == "checkpoints" \
                and {"state.json", "trainable.pt", "optimizer.pt"} <= set(files):
            try:
                st = read_json(os.path.join(root, "state.json"))
            except (OSError, json.JSONDecodeError):
                continue
            if st.get("fingerprint") == fingerprint:
                cand = (run_progress(st), os.path.dirname(os.path.dirname(root)))
                if best is None or cand > best:
                    best = cand
    return best


# --------------------------------------------------------------------------------------------------------------
# Data: manifest, split, class weights, epoch order, schedule (pure Python, no torch)
# --------------------------------------------------------------------------------------------------------------
def band(votes):
    """Agreement band, identical to run_experiment.band."""
    if votes in (0, 7):
        return "unanimous"
    if votes in (3, 4):
        return "borderline"
    return "strong"


def stratum_of(row):
    return f"{row['label']}|{band(row['ssa_votes'])}"


def read_manifest(path):
    """train_manifest.csv -> rows [{'image', 'label', 'ssa_votes'}], sorted by image. POOL tiles only."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        missing = {"image", "label", "ssa_votes"} - fields
        if missing:
            die(f"{path}: missing column(s) {sorted(missing)}; expected image,label,ssa_votes")
        raw = list(reader)
    rows, seen = [], set()
    for i, r in enumerate(raw, 2):
        name = (r["image"] or "").strip()
        if not name or os.path.basename(name) != name or not name.lower().endswith(".png"):
            die(f"{path} line {i}: bad image name {name!r}")
        if name in seen:
            die(f"{path} line {i}: duplicate image {name}")
        seen.add(name)
        label = (r["label"] or "").strip()
        if label not in CLASSES:
            die(f"{path} line {i}: label {label!r} is not one of {CLASSES}")
        try:
            votes = int(r["ssa_votes"])
        except (TypeError, ValueError):
            die(f"{path} line {i}: ssa_votes {r['ssa_votes']!r} is not an integer")
        if not 0 <= votes <= 7:
            die(f"{path} line {i}: ssa_votes {votes} outside 0..7")
        if (label == "SSA") != (votes >= 4):
            die(f"{path} line {i}: label {label} contradicts {votes}/7 SSA votes (majority vote)")
        for col in ("partition", "Partition"):
            if col in fields and (r[col] or "").strip().lower() != "train":
                die(f"{path} line {i}: partition {r[col]!r} -- only training-pool tiles may be in this manifest")
        rows.append({"image": name, "label": label, "ssa_votes": votes})
    if not rows:
        die(f"{path}: no rows")
    rows.sort(key=lambda r: r["image"])
    return rows


def pool_sha256(names):
    return sha256_bytes("\n".join(sorted(names)).encode())


def hash_order(names, seed, salt):
    """A seeded permutation that does not depend on any RNG implementation: sort by sha256(seed|salt|name)."""
    return sorted(names, key=lambda n: (hashlib.sha256(f"{seed}|{salt}|{n}".encode()).hexdigest(), n))


def make_split(rows, seed=SEED, val_fraction=VAL_FRACTION):
    """Validation slice stratified by label x agreement band.

    Per stratum the quota is n * fraction; quotas are floored and the remaining seats go to the largest
    remainders (ties: stratum name), so the total is round(N * fraction). Within a stratum the validation tiles
    are the first ones in hash_order(seed, 'split'). Exact integer arithmetic, no RNG.
    """
    val_fraction = Fraction(val_fraction)
    by = {}
    for r in rows:
        by.setdefault(stratum_of(r), []).append(r["image"])
    n_total = len(rows)
    target = int(math.floor(n_total * val_fraction + Fraction(1, 2)))
    quota = {k: len(v) * val_fraction for k, v in by.items()}
    n_val = {k: int(math.floor(q)) for k, q in quota.items()}
    spare = target - sum(n_val.values())
    for k in sorted(by, key=lambda k: (-(quota[k] - n_val[k]), k))[:spare]:
        n_val[k] += 1
    val, train, strata = [], [], {}
    for k in sorted(by):
        order = hash_order(by[k], seed, "split")
        val += order[: n_val[k]]
        train += order[n_val[k]:]
        strata[k] = {"n": len(by[k]), "n_val": n_val[k], "n_train": len(by[k]) - n_val[k]}
    return {"seed": seed, "val_fraction": float(val_fraction), "n_pool": n_total,
            "method": "stratified by label x agreement band (0/7 unanimous, 3/4 borderline, else strong); "
                      "largest-remainder quotas; within a stratum the first tiles by sha256(seed|split|name)",
            "strata": strata, "train": sorted(train), "val": sorted(val)}


def class_weights(train_rows):
    """Inverse class frequency in the training split, scaled so the mean weight over the split is 1."""
    counts = {c: sum(r["label"] == c for r in train_rows) for c in CLASSES}
    if min(counts.values()) == 0:
        die(f"a class is missing from the training split: {counts}")
    n = len(train_rows)
    return {c: n / (len(CLASSES) * counts[c]) for c in CLASSES}, counts


def epoch_groups(train_names, seed, epoch, effective_batch=EFFECTIVE_BATCH):
    """Accumulation groups (one optimiser step each) for a 0-based epoch; the order is a pure function of
    (seed, epoch), so a resumed run sees exactly the groups an uninterrupted run would."""
    order = hash_order(train_names, seed, f"epoch{epoch}")
    return [order[i:i + effective_batch] for i in range(0, len(order), effective_batch)]


def warmup_steps_for(total_steps):
    return int(math.ceil(total_steps * WARMUP_FRACTION))


def lr_at(step, total_steps, warmup_steps, base_lr=LEARNING_RATE):
    """Learning rate of the optimiser step with 0-based index `step`: linear warm-up then half-cosine to zero.
    Same formula as transformers.get_cosine_schedule_with_warmup (so step 0 has lr 0, as in the HF Trainer)."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def val_metrics(records):
    """records: [{'label', 'pred', 'score_HP', 'score_SSA'}]. Balanced accuracy = mean of the two recalls."""
    out = {"n": len(records)}
    recalls = []
    for c in CLASSES:
        sub = [r for r in records if r["label"] == c]
        rec = sum(r["pred"] == c for r in sub) / max(len(sub), 1)
        out[f"n_{c}"] = len(sub)
        out[f"recall_{c}"] = round(rec, 6)
        recalls.append(rec)
    out["balanced_accuracy"] = round(sum(recalls) / len(recalls), 6)
    out["accuracy"] = round(sum(r["pred"] == r["label"] for r in records) / max(len(records), 1), 6)
    out["ssa_call_rate"] = round(sum(r["pred"] == "SSA" for r in records) / max(len(records), 1), 6)
    # two-way log loss: HP-vs-SSA softmax over the two label scores (lower is better); per class, then averaged
    nll = {c: [] for c in CLASSES}
    for r in records:
        m = max(r["score_HP"], r["score_SSA"])
        z = m + math.log(math.exp(r["score_HP"] - m) + math.exp(r["score_SSA"] - m))
        nll[r["label"]].append(z - r[f"score_{r['label']}"])
    per_class = [sum(v) / len(v) for v in nll.values() if v]
    out["two_way_nll_class_balanced"] = round(sum(per_class) / max(len(per_class), 1), 6)
    out["label_mass_mean"] = round(sum(min(1.0, math.exp(r["score_HP"]) + math.exp(r["score_SSA"]))
                                       for r in records) / max(len(records), 1), 6)
    # REPORT ONLY, never used for selection. Tier 1 samples the label at temperature 1.0 with top_p 0.95, so its
    # expected accuracy is the mean probability of the true label, not the argmax accuracy above. Per tile:
    # p = two-way softmax probability of the true label; top_p 0.95 removes a label below 5%, so p >= 0.95 counts
    # as 1 and p <= 0.05 as 0. It ignores the mass on non-label tokens (label_mass_mean) and it is measured on
    # the validation slice, not on dev: a guide to how far tier 1 can fall below the argmax figure, no more.
    p_true = {c: [] for c in CLASSES}
    for r in records:
        m = max(r["score_HP"], r["score_SSA"])
        z = m + math.log(math.exp(r["score_HP"] - m) + math.exp(r["score_SSA"] - m))
        prob = math.exp(r[f"score_{r['label']}"] - z)
        p_true[r["label"]].append(1.0 if prob >= 0.95 else (0.0 if prob <= 0.05 else prob))
    flat = [v for c in CLASSES for v in p_true[c]]
    per_class = [sum(v) / len(v) for v in p_true.values() if v]
    out["expected_tier1_accuracy"] = round(sum(flat) / max(len(flat), 1), 6)
    out["expected_tier1_balanced_accuracy"] = round(sum(per_class) / max(len(per_class), 1), 6)
    return out


# --------------------------------------------------------------------------------------------------------------
# Bundle checks
# --------------------------------------------------------------------------------------------------------------
def find_tile(gridded_dir, image):
    stem = image[:-4] if image.lower().endswith(".png") else image
    for cand in (image, stem + ".png", image + ".png", stem + "_grid.png"):
        p = os.path.join(gridded_dir, cand)
        if os.path.isfile(p):
            return p
    return None


def load_bundle(data_dir, allow_pool_mismatch=False):
    """Read and verify the training bundle. Returns a dict. Only the tiles named in train_manifest.csv are
    ever opened: other files in gridded/ (the shared bundle also holds the dev tiles, for the later evaluation)
    are counted and left alone."""
    from PIL import Image

    rows = read_manifest(os.path.join(data_dir, "train_manifest.csv"))
    names = [r["image"] for r in rows]
    got_sha = pool_sha256(names)
    pool_verified = len(names) == EXPECTED_POOL_N and got_sha == EXPECTED_POOL_SHA256
    if not pool_verified and not allow_pool_mismatch:
        die(f"train_manifest.csv lists {len(names)} tiles with name-list sha256 {got_sha[:16]}..., but the frozen "
            f"fewshot_pool has {EXPECTED_POOL_N} tiles and sha256 {EXPECTED_POOL_SHA256[:16]}.... The manifest must "
            "contain exactly the pool tiles (train minus dev). --allow-pool-mismatch exists for smoke tests only.")
    excluded_path = os.path.join(data_dir, "excluded_images.txt")
    if os.path.exists(excluded_path):
        excluded = {l.strip() for l in read_bytes(excluded_path).decode("utf-8").splitlines() if l.strip()}
        clash = sorted(excluded & set(names))
        if clash:
            die(f"{len(clash)} manifest tiles are listed in excluded_images.txt (dev/test), e.g. {clash[:5]}")
    dev_path = os.path.join(data_dir, "dev_tiles.csv")   # the shared bundle also carries the dev tiles (no labels)
    if os.path.exists(dev_path):
        with open(dev_path, newline="") as fh:
            dev_names = {(r.get("image") or "").strip() for r in csv.DictReader(fh)}
        clash = sorted(dev_names & set(names))
        if clash:
            die(f"{len(clash)} manifest tiles are dev tiles according to dev_tiles.csv, e.g. {clash[:5]}")
    prompt_path = os.path.join(data_dir, "prompts", "cte_p1.txt")
    if not os.path.isfile(prompt_path):
        die(f"missing {prompt_path}")
    prompt_bytes = read_bytes(prompt_path)
    if sha256_bytes(prompt_bytes) != EXPECTED_PROMPT_SHA256:
        die(f"{prompt_path} has sha256 {sha256_bytes(prompt_bytes)[:16]}..., expected "
            f"{EXPECTED_PROMPT_SHA256[:16]}... (prompts/rendered/cte_p1.txt of the pipeline). The prompt is frozen.")
    gridded_dir = os.path.join(data_dir, "gridded")
    if not os.path.isdir(gridded_dir):
        die(f"missing {gridded_dir}")
    paths, digests = {}, []
    for name in names:
        p = find_tile(gridded_dir, name)
        if p is None:
            die(f"no gridded PNG for {name} in {gridded_dir}")
        with Image.open(p) as im:
            if im.size != TILE_SIZE:
                die(f"{p}: size {im.size}, expected {TILE_SIZE}")
        paths[name] = p
        digests.append(f"{name}:{sha256_file(p)}")
    used = {os.path.basename(p) for p in paths.values()}
    unlisted = sorted(f for f in os.listdir(gridded_dir) if f not in used and not f.startswith("."))
    return {"rows": rows, "paths": paths, "prompt_text": prompt_bytes.decode("utf-8"),
            "prompt_sha256": sha256_bytes(prompt_bytes), "pool_sha256": got_sha, "pool_verified": pool_verified,
            "manifest_sha256": sha256_file(os.path.join(data_dir, "train_manifest.csv")),
            "gridded_sha256": sha256_bytes("\n".join(digests).encode()),
            "n_unlisted_files_in_gridded": len(unlisted)}


def model_fingerprint(model_dir, hash_weights, allow_mismatch=False, cache_path=None):
    """Identify the attached weights: sha256 of the small files, sizes (and optionally sha256) of the shards."""
    small, problems = {}, []
    for f, want in EXPECTED_MODEL_FILES.items():
        p = os.path.join(model_dir, f)
        if not os.path.isfile(p):
            small[f] = None
            problems.append(f"{f} missing")
            continue
        small[f] = sha256_file(p)
        if small[f] != want:
            problems.append(f"{f} sha256 {small[f][:12]}... != {want[:12]}...")
    shards = {}
    for f, want in EXPECTED_MODEL_SHARD_BYTES.items():
        p = os.path.join(model_dir, f)
        size = os.path.getsize(p) if os.path.isfile(p) else None
        shards[f] = {"bytes": size}
        if size != want:
            problems.append(f"{f} is {size} bytes, expected {want}")
    verified = not problems
    if problems and not allow_mismatch:
        die(f"{model_dir} is not {MODEL_ID} @ {MODEL_REVISION[:8]}: " + "; ".join(problems) +
            ". Attach the official files unchanged (--allow-model-mismatch exists for local smoke tests only).")
    if hash_weights and verified:
        cached = {}
        if cache_path and os.path.exists(cache_path):
            try:
                cached = read_json(cache_path)
            except (OSError, json.JSONDecodeError):
                cached = {}
        for f in shards:
            if cached.get(f, {}).get("bytes") == shards[f]["bytes"] and cached[f].get("sha256"):
                shards[f]["sha256"] = cached[f]["sha256"]
            else:
                log(f"hashing {f} ...")
                shards[f]["sha256"] = sha256_file(os.path.join(model_dir, f), chunk=1 << 24)
        if cache_path:
            atomic_write_json(cache_path, shards)
    return {"model_id": MODEL_ID, "revision_expected": MODEL_REVISION, "small_files_sha256": small,
            "weight_shards": shards, "matches_frozen_revision_files": verified, "path": model_dir}


# --------------------------------------------------------------------------------------------------------------
# Official chat format and targets (needs the processor, not the weights)
# --------------------------------------------------------------------------------------------------------------
def user_messages(prompt_text, image=None):
    """One user turn, image first, then the prompt; no system prompt (PLAN step-3 addendum, amendment 2)."""
    image_part = {"type": "image"} if image is None else {"type": "image", "image": image}
    return [{"role": "user", "content": [image_part, {"type": "text", "text": prompt_text}]}]


def load_processor(model_dir, image_backend="default"):
    """The model's own processor. 'default' is exactly AutoProcessor.from_pretrained(model_dir), which is also
    what the evaluation script (infer_jobs.py) loads, so training and evaluation preprocess pixels the same way
    on the same machine image. 'pil' pins the PIL image pipeline instead (bilinear 224->896, rescale, normalise).
    The class that was used is recorded in train_summary.json and best_adapter/step4_meta.json."""
    import transformers
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    if image_backend == "pil":
        from transformers import AutoImageProcessor
        kw = {"backend": "pil"} if parse_version(transformers.__version__) >= (5,) else {"use_fast": False}
        processor.image_processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, **kw)
    return processor


def label_scoring_plan(target_ids):
    """How to read the label from the logits, given the real tokenisation of the two targets.

    The targets share a prefix ('{\\n  "label": "'); the label is whatever follows. If each label is ONE token
    (true for MedGemma's tokenizer: 'HP' and 'SSA' are single tokens and the closing quote is a separate one),
    the comparison is the two logits at the label position. Otherwise each candidate is scored as an exact
    teacher-forced sequence log-probability over everything after the shared prefix (label tokens through the
    closing quote, so candidates of different length are compared as complete, terminated strings).
    """
    ids = [list(target_ids[c]) for c in CLASSES]
    k = 0
    while k < min(len(x) for x in ids) and len({x[k] for x in ids}) == 1:
        k += 1
    rest = {c: x[k:] for c, x in zip(CLASSES, ids)}
    if any(len(v) == 0 for v in rest.values()):
        die(f"one target is a token-prefix of the other, cannot compare: {target_ids}")
    s = 0
    while s < min(len(v) for v in rest.values()) - 1 and len({v[-1 - s] for v in rest.values()}) == 1:
        s += 1
    label_only = {c: (v[: len(v) - s] if s else v) for c, v in rest.items()}
    single = all(len(v) == 1 for v in label_only.values())
    return {"common_prefix_ids": ids[0][:k], "common_suffix_len": s,
            "mode": "single_token_logits" if single else "sequence_log_prob",
            "scored_ids": label_only if single else rest, "label_token_ids": label_only}


class ExampleBuilder:
    """Builds token ids / labels / pixel values for a tile in the official chat format.

    prompt = processor.apply_chat_template([user: image, cte_p1 text], add_generation_prompt=True), expanded and
    tokenised by the processor itself (add_special_tokens=False: the template already carries <bos>);
    example = prompt tokens + target tokens; labels = -100 on every prompt token (image tokens included).
    """

    def __init__(self, processor, prompt_text):
        import torch

        self.torch = torch
        self.processor = processor
        self.tok = processor.tokenizer
        self.prompt_text = prompt_text
        self.template_text = processor.apply_chat_template(user_messages(prompt_text), add_generation_prompt=True,
                                                           tokenize=False)
        boi = getattr(self.tok, "boi_token", None) or "<start_of_image>"
        if self.template_text.count(boi) != 1 or not self.template_text.endswith("<start_of_turn>model\n") \
                or prompt_text.strip() not in self.template_text or "<start_of_turn>system" in self.template_text:
            die("the chat template did not render as one user turn (image, then prompt) plus the generation prompt: "
                + repr(self.template_text[:120]) + " ... " + repr(self.template_text[-60:]))
        if self.template_text.index(boi) > self.template_text.index(prompt_text.strip()):
            die("the image placeholder does not precede the prompt text in the rendered template")
        self.image_token_id = self.tok.convert_tokens_to_ids("<image_soft_token>")
        if self.image_token_id is None or self.image_token_id == getattr(self.tok, "unk_token_id", -1):
            die("the tokenizer has no <image_soft_token>")
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.target_text = {c: TARGET_TEMPLATE.format(label=c) for c in CLASSES}
        self.target_ids = {c: [int(i) for i in self.tok(self.target_text[c], add_special_tokens=False)["input_ids"]]
                           for c in CLASSES}
        for c in CLASSES:
            if self.tok.decode(self.target_ids[c]) != self.target_text[c]:
                die(f"target for {c} does not round-trip through the tokenizer: {self.target_ids[c]}")
        self.scoring = label_scoring_plan(self.target_ids)
        self.ref_prompt_ids = None
        self._cache = {}

    # -- image side --------------------------------------------------------------------------------------------
    def encode_prompt(self, image):
        """Full processor call for one tile -> (prompt input_ids, pixel_values[3,H,W]); verifies the layout."""
        torch = self.torch
        enc = self.processor(text=[self.template_text], images=[[image.convert("RGB")]], return_tensors="pt",
                             add_special_tokens=False)
        ids = enc["input_ids"][0]
        n_img = int((ids == self.image_token_id).sum())
        if n_img != EXPECTED_IMAGE_TOKENS:
            die(f"{n_img} image tokens in the prompt, expected {EXPECTED_IMAGE_TOKENS}")
        if "token_type_ids" in enc and not torch.equal(enc["token_type_ids"][0] != 0, ids == self.image_token_id):
            die("processor token_type_ids do not mark exactly the image tokens")
        if self.ref_prompt_ids is None:
            self.ref_prompt_ids = ids.clone()
        elif not torch.equal(ids, self.ref_prompt_ids):
            die("prompt token ids changed between tiles; they must be identical for every tile")
        pix = enc["pixel_values"]
        if pix.dim() != 4 or pix.shape[0] != 1:
            die(f"unexpected pixel_values shape {tuple(pix.shape)} (pan-and-scan must be off)")
        return ids, pix[0]

    def encode_image(self, image):
        return self.encode_prompt(image)[1]

    def official_one_step(self, image):
        """processor.apply_chat_template(..., tokenize=True) for the same image + text: the reference encoding."""
        return self.processor.apply_chat_template(user_messages(self.prompt_text, image.convert("RGB")),
                                                  add_generation_prompt=True, tokenize=True, return_dict=True,
                                                  return_tensors="pt")

    # -- token side --------------------------------------------------------------------------------------------
    def training_ids(self, label):
        """input_ids / labels / token_type_ids / attention_mask for one class (identical for every tile)."""
        torch = self.torch
        if self.ref_prompt_ids is None:
            die("encode one tile first (the prompt ids come from the processor)")
        if label not in self._cache:
            t = torch.tensor(self.target_ids[label], dtype=torch.long)
            ids = torch.cat([self.ref_prompt_ids, t])
            labels = torch.full_like(ids, -100)
            labels[len(self.ref_prompt_ids):] = t
            self._cache[label] = {"input_ids": ids, "labels": labels,
                                  "token_type_ids": (ids == self.image_token_id).long(),
                                  "attention_mask": torch.ones_like(ids)}
        return self._cache[label]

    def training_example(self, image, label):
        """One complete training example (used by the tests and by the pixel path)."""
        _, pix = self.encode_prompt(image)
        ex = dict(self.training_ids(label))
        ex["pixel_values"] = pix
        return ex

    def scoring_ids(self, context=()):
        """Prompt + the shared target prefix (+ a teacher-forced context): the next token is the label."""
        torch = self.torch
        key = ("score", tuple(context))
        if key not in self._cache:
            extra = torch.tensor(list(self.scoring["common_prefix_ids"]) + list(context), dtype=torch.long)
            ids = torch.cat([self.ref_prompt_ids, extra])
            self._cache[key] = {"input_ids": ids, "token_type_ids": (ids == self.image_token_id).long(),
                                "attention_mask": torch.ones_like(ids)}
        return self._cache[key]

    def describe(self):
        t = self.tok
        return {"prompt_tokens": None if self.ref_prompt_ids is None else int(len(self.ref_prompt_ids)),
                "image_token_id": int(self.image_token_id),
                "target_text": self.target_text, "target_ids": self.target_ids,
                "target_tokens": {c: t.convert_ids_to_tokens(v) for c, v in self.target_ids.items()},
                "scoring_mode": self.scoring["mode"],
                "common_prefix_tokens": t.convert_ids_to_tokens(self.scoring["common_prefix_ids"]),
                "scored_ids": self.scoring["scored_ids"],
                "scored_tokens": {c: t.convert_ids_to_tokens(v) for c, v in self.scoring["scored_ids"].items()}}


def collate(examples, pad_id, torch):
    """Right-pad a list of example dicts (Google's notebook pads on the right for training). With this prompt
    and these two targets every sequence has the same length, so nothing is padded in practice."""
    n = max(len(e["input_ids"]) for e in examples)
    out = {"input_ids": torch.full((len(examples), n), pad_id, dtype=torch.long),
           "attention_mask": torch.zeros((len(examples), n), dtype=torch.long),
           "token_type_ids": torch.zeros((len(examples), n), dtype=torch.long)}
    if "labels" in examples[0]:
        out["labels"] = torch.full((len(examples), n), -100, dtype=torch.long)
    for i, e in enumerate(examples):
        m = len(e["input_ids"])
        out["input_ids"][i, :m] = e["input_ids"]
        out["attention_mask"][i, :m] = e["attention_mask"]
        out["token_type_ids"][i, :m] = e["token_type_ids"]
        if "labels" in out:
            out["labels"][i, :m] = e["labels"]
    for key in ("pixel_values", "image_features"):
        if key in examples[0]:
            out[key] = torch.stack([e[key] for e in examples])
    return out


def trailing_supervised(batch):
    """T if every row is unpadded and its supervised tokens are exactly its last T tokens (fast path), else 0."""
    labels = batch["labels"]
    if int(batch["attention_mask"].min()) == 0:
        return 0
    sup = labels != -100
    t = int(sup[0].sum())
    if t == 0 or t >= labels.shape[1]:
        return 0
    if bool(sup[:, -t:].all()) and not bool(sup[:, :-t].any()):
        return t
    return 0


def ce_trailing(torch, logits_kept, targets):
    """Per-example mean cross-entropy when the supervised tokens are the last T ones.
    logits_kept: [B, T+1, V], the logits at the last T+1 positions; targets: [B, T], the last T token ids."""
    b, t = targets.shape
    ce = torch.nn.functional.cross_entropy(logits_kept[:, :-1, :].reshape(b * t, -1).float(), targets.reshape(-1),
                                           reduction="none")
    return ce.view(b, t).mean(dim=1)


def ce_general(torch, logits, labels):
    """Per-example mean cross-entropy over the supervised positions (labels != -100), any layout."""
    shift_logits, shift_labels = logits[:, :-1, :], labels[:, 1:]
    b, n, v = shift_logits.shape
    ce = torch.nn.functional.cross_entropy(shift_logits.reshape(b * n, v).float(), shift_labels.reshape(-1),
                                           ignore_index=-100, reduction="none").view(b, n)
    return ce.sum(dim=1) / (shift_labels != -100).sum(dim=1).clamp(min=1)


# --------------------------------------------------------------------------------------------------------------
# Environment, precision, model, LoRA
# --------------------------------------------------------------------------------------------------------------
def package_versions(need_bnb, need_accelerate):
    """Import what the run needs. Returns (versions, problem_message or None); the message lists everything
    that is missing or too old and how to install it in a Kaggle notebook."""
    import importlib

    wanted = ["torch", "transformers", "peft", "safetensors"] + (["accelerate"] if need_accelerate else []) \
        + (["bitsandbytes"] if need_bnb else [])
    versions, problems = {"python": sys.version.split()[0]}, []
    for name in wanted:
        try:
            mod = importlib.import_module(name)
            versions[name] = getattr(mod, "__version__", "unknown")
            if parse_version(versions[name]) < parse_version(MIN_VERSIONS[name]):
                problems.append(f"{name} {versions[name]} is older than the required {MIN_VERSIONS[name]}")
        except Exception as e:   # ImportError, or an incompatible pair (e.g. an old peft with a new transformers)
            versions[name] = None
            problems.append(f"{name} >= {MIN_VERSIONS[name]} cannot be imported ({type(e).__name__}: {str(e)[:160]})")
    for name in ("accelerate", "bitsandbytes", "tokenizers", "numpy", "PIL", "sentencepiece"):
        if name not in versions:
            try:
                versions[name] = getattr(importlib.import_module(name), "__version__", "unknown")
            except Exception:
                versions[name] = None
    message = None
    if problems:
        message = ("package check failed:\n  - " + "\n  - ".join(problems) +
                   "\nThe Kaggle image must provide these. With internet ON:"
                   "\n  pip install -q -U peft bitsandbytes accelerate"
                   "\nWith internet OFF: attach a private dataset of wheels and run"
                   "\n  pip install --no-index --find-links /kaggle/input/<wheels-slug> peft bitsandbytes accelerate"
                   "\n(bitsandbytes is only needed when the GPU has no native bf16, i.e. on Kaggle's T4.)")
    return versions, message


def bf16_probe(torch, device):
    """Does bf16 arithmetic actually run on this GPU (older GPUs emulate it)? Returns (ok, detail)."""
    try:
        g = torch.Generator(device="cpu").manual_seed(0)
        a = torch.randn(2, 4, 64, 32, generator=g).to(device)
        b = torch.randn(2, 4, 32, 64, generator=g).to(device)
        ref = torch.matmul(a, b)
        got = torch.matmul(a.to(torch.bfloat16), b.to(torch.bfloat16)).float()
        got2 = torch.bmm(a[0].to(torch.bfloat16), b[0].to(torch.bfloat16)).float()
        lin = torch.nn.functional.linear(a.to(torch.bfloat16), b[0, 0].t().contiguous().to(torch.bfloat16)).float()
        sm = torch.softmax(got.to(torch.bfloat16), dim=-1).float()
        ok = bool(torch.isfinite(got).all() and torch.isfinite(got2).all() and torch.isfinite(lin).all()
                  and torch.isfinite(sm).all() and (got - ref).abs().max() < 0.5)
        return ok, "bf16 matmul/bmm/linear/softmax ran" + ("" if ok else " but gave wrong or non-finite values")
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}"


def resolve_precision(args, torch):
    """Decide weights dtype / quantisation. 'auto' follows Google's notebook test for bf16 (compute capability
    >= 8); on older GPUs (Kaggle's T4) it loads 4-bit nf4 with float32 compute, because bf16 there is
    emulated and fp16 overflows in Gemma 3."""
    if args.device == "cpu":
        if args.precision not in ("auto", "float32"):
            die("--device cpu only supports --precision float32 (local tests)")
        return {"mode": "float32", "weights_dtype": "float32", "compute_dtype": "float32", "autocast": False,
                "used_4bit": False, "reason": "CPU run (tests only)", "gpu": None}
    if not torch.cuda.is_available():
        die("no CUDA GPU visible. In the Kaggle notebook choose Settings > Accelerator > GPU T4 x2.")
    dev = torch.device("cuda:0")
    prop = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    native = cap[0] >= 8
    gpu = {"name": prop.name, "compute_capability": f"{cap[0]}.{cap[1]}", "memory_gb": round(prop.total_memory / 2**30, 2),
           "n_gpus_visible": torch.cuda.device_count(), "native_bf16": native, "cuda": torch.version.cuda}
    mode, reason = args.precision, f"--precision {args.precision}"
    if mode == "auto":
        if native and gpu["memory_gb"] >= 14:
            mode, reason = "bf16", f"auto: {prop.name} has native bf16 (capability {gpu['compute_capability']})"
        elif native:
            mode, reason = "4bit", f"auto: native bf16 but only {gpu['memory_gb']} GB, 4-bit needed for memory"
        else:
            mode = "4bit"
            reason = (f"auto: {prop.name} (capability {gpu['compute_capability']}) has no native bf16 "
                      "(Google's notebook requires capability >= 8), so 4-bit nf4 is used")
    out = {"mode": mode, "used_4bit": mode == "4bit", "reason": reason, "gpu": gpu}
    if mode == "bf16":
        if not native:
            ok, detail = bf16_probe(torch, dev)
            gpu["bf16_emulation_probe"] = detail
            if not ok:
                die(f"--precision bf16 was forced but bf16 does not work on {prop.name}: {detail}")
            out["reason"] += " (bf16 is EMULATED on this GPU: correct but possibly slow)"
        out.update({"weights_dtype": "bfloat16", "compute_dtype": "bfloat16", "autocast": True})
    elif mode == "4bit":
        cd = args.compute_dtype
        if cd == "auto":
            cd = "bfloat16" if native else "float32"
        elif cd == "bf16":
            cd = "bfloat16"
        if cd == "bfloat16" and not native:
            ok, detail = bf16_probe(torch, dev)
            gpu["bf16_emulation_probe"] = detail
            if not ok:
                die(f"--compute-dtype bf16 was forced but bf16 does not work on {prop.name}: {detail}")
        out.update({"weights_dtype": "4-bit nf4 (double quantisation); unquantised modules float32",
                    "compute_dtype": cd, "autocast": cd == "bfloat16",
                    "quantized_scope": QUANTIZED_SCOPE, "vision_tower": "float32, not quantised",
                    "bitsandbytes_config": {"load_in_4bit": True, "bnb_4bit_use_double_quant": True,
                                            "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": cd,
                                            "llm_int8_skip_modules": list(BNB_SKIP_MODULES)}})
    elif mode == "float32":
        out.update({"weights_dtype": "float32", "compute_dtype": "float32", "autocast": False})
    else:
        die(f"unknown precision {mode}")
    return out


def load_model(args, model_dir, prec):
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText

    dtype_key = "dtype" if parse_version(transformers.__version__) >= (5,) else "torch_dtype"
    kw = {"attn_implementation": args.attn_implementation, "local_files_only": True}
    on_gpu = args.device == "cuda"
    device_map = {"": 0} if args.device_map == "single" else args.device_map
    if prec["mode"] == "4bit":
        from transformers import BitsAndBytesConfig

        cd = getattr(torch, prec["compute_dtype"])
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                                       bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=cd,
                                                       llm_int8_skip_modules=list(BNB_SKIP_MODULES))
        kw[dtype_key] = cd
        kw["device_map"] = device_map
    else:
        kw[dtype_key] = getattr(torch, prec["weights_dtype"])
        if on_gpu:
            kw["device_map"] = device_map
    model = AutoModelForImageTextToText.from_pretrained(model_dir, **kw)
    if type(model).__name__ != "Gemma3ForConditionalGeneration":
        die(f"loaded a {type(model).__name__}, expected Gemma3ForConditionalGeneration")
    if prec["mode"] == "4bit":
        check_quantized_scope(model)
    return model


def check_quantized_scope(model):
    """After a 4-bit load: bitsandbytes must have replaced exactly the language model's linear layers (7 per
    block) and nothing else. In particular not the frozen SigLIP vision tower: memory does not require that, and
    uncalibrated 4-bit weights in the image encoder could blur the detail HP vs SSA depends on. Returns the
    number of 4-bit modules; stops the run if the installed transformers / bitsandbytes did something else."""
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
            f"{len(LM_LINEAR_LEAVES)} x {n_layers} = {len(LM_LINEAR_LEAVES) * n_layers}, all inside the language model; "
            f"{len(outside)} lie outside it, e.g. {outside[:3]}. BNB_SKIP_MODULES does not match this version's module "
            "names: fix the list (or install the transformers version the probe was run with) before training.")
    return len(names)


def find_submodule(model, leaf):
    hits = [(n, m) for n, m in model.named_modules() if n.split(".")[-1] == leaf]
    if not hits:
        die(f"the model has no submodule named {leaf}")
    return min(hits, key=lambda h: len(h[0]))


def select_lora_targets(model):
    """Full names of every nn.Linear inside the language model (bitsandbytes' Linear4bit is an nn.Linear).

    Excluded on purpose: anything under vision_tower or multi_modal_projector (the vision side is frozen and not
    adapted) and the output head lm_head (tied to the embedding matrix; not a LoRA target in Google's notebook
    either, where PEFT's "all-linear" skips the output layer). Returns (targets, not_targeted_linear_names).
    """
    import torch.nn as nn

    out_emb = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    targets, others = [], []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        parts = name.split(".")
        if ("language_model" in parts and "vision_tower" not in parts and "multi_modal_projector" not in parts
                and parts[-1] != "lm_head" and mod is not out_emb):
            targets.append(name)
        else:
            others.append(name)
    return targets, others


def check_lora_targets(model, targets):
    """The Gemma 3 text model has exactly 7 linear layers per block (q,k,v,o and the three MLP projections)."""
    n_layers = model.config.text_config.num_hidden_layers
    bad = [t for t in targets if t.split(".")[-1] not in LM_LINEAR_LEAVES]
    if bad or len(targets) != len(LM_LINEAR_LEAVES) * n_layers:
        die(f"unexpected language-model linear layers: {len(targets)} found, expected "
            f"{len(LM_LINEAR_LEAVES)} x {n_layers}; unexpected names: {bad[:8]}")
    if any("vision_tower" in t.split(".") or "multi_modal_projector" in t.split(".") for t in targets):
        die("a vision module ended up in the LoRA target list")


def attach_lora(model, targets, torch):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
                     target_modules=list(targets), task_type="CAUSAL_LM")
    torch.manual_seed(SEED)   # LoRA A is randomly initialised (B = 0)
    return get_peft_model(model, cfg)


def verify_lora(peft_model, targets):
    """After wrapping: LoRA sits on exactly the target modules, only LoRA matrices are trainable, and nothing
    under the vision tower or the projector is wrapped or trainable."""
    prefix = "base_model.model."
    wrapped = sorted(n[len(prefix):] if n.startswith(prefix) else n
                     for n, m in peft_model.named_modules() if hasattr(m, "lora_A") and hasattr(m, "lora_B"))
    if wrapped != sorted(targets):
        extra = sorted(set(wrapped) - set(targets))[:5]
        missing = sorted(set(targets) - set(wrapped))[:5]
        die(f"LoRA wrapped {len(wrapped)} modules, expected {len(targets)}; extra {extra}, missing {missing}")
    trainable = [(n, p) for n, p in peft_model.named_parameters() if p.requires_grad]
    for n, _ in trainable:
        parts = n.split(".")
        if "lora_A" not in parts and "lora_B" not in parts:
            die(f"a non-LoRA parameter is trainable: {n}")
        if "language_model" not in parts or "vision_tower" in parts or "multi_modal_projector" in parts:
            die(f"a trainable parameter lies outside the language model: {n}")
    for n, m in peft_model.named_modules():
        parts = n.split(".")
        if ("vision_tower" in parts or "multi_modal_projector" in parts) and (
                "lora" in parts[-1].lower() or hasattr(m, "lora_A")):
            die(f"a LoRA module was created on the vision side: {n}")
    if len(trainable) != 2 * len(targets):
        die(f"{len(trainable)} trainable tensors, expected {2 * len(targets)} (A and B per target)")
    n_train = sum(p.numel() for _, p in trainable)
    n_all = sum(p.numel() for p in peft_model.parameters())
    return {"n_target_modules": len(targets), "trainable_parameters": n_train, "all_parameters_as_stored": n_all}


class Engine:
    """Forward passes. Two equivalent ways to feed the image:
      pixel path  : input_ids + pixel_values, the model runs the (frozen) vision tower itself;
      cached path : the frozen vision tower + projector are run once per tile, and their 256 x hidden output is
                    scattered into the input embeddings exactly as the model's own forward does.
    The cached path is checked against the pixel path on the real model before it is used."""

    def __init__(self, torch, model, prec, image_token_id):
        self.torch, self.model, self.prec, self.image_token_id = torch, model, prec, image_token_id
        self.embed = model.get_input_embeddings()
        self.device = self.embed.weight.device
        self.embed_dtype = self.embed.weight.dtype
        self.vocab_size = model.config.text_config.vocab_size
        self.vision_name, self.vision_tower = find_submodule(model, "vision_tower")
        self.vision_dtype = next((p.dtype for p in self.vision_tower.parameters()
                                  if p.dtype.is_floating_point and type(p).__name__ != "Params4bit"), torch.float32)
        self.has_keep = "logits_to_keep" in inspect.signature(model.forward).parameters

    def autocast(self):
        if self.prec["autocast"]:
            return self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16)
        return contextlib.nullcontext()

    def image_features(self, pixel_values):
        """[B,3,H,W] -> [B,256,hidden] in the embedding dtype (what the model's forward scatters)."""
        torch = self.torch
        with torch.no_grad(), self.autocast():
            out = self.model.get_image_features(pixel_values.to(self.device, self.vision_dtype))
        feats = out if torch.is_tensor(out) else getattr(out, "pooler_output", None)
        if feats is None or feats.dim() != 3 or feats.shape[1] != EXPECTED_IMAGE_TOKENS:
            raise RuntimeError(f"unexpected get_image_features output: {type(out).__name__}")
        return feats.to(self.embed_dtype)

    def forward_logits(self, batch, keep=0):
        """Logits [B, keep or L, V] (float32). batch holds input_ids, attention_mask, token_type_ids and either
        pixel_values or image_features."""
        torch = self.torch
        ids = batch["input_ids"].to(self.device)
        kw = {"attention_mask": batch["attention_mask"].to(self.device),
              "token_type_ids": batch["token_type_ids"].to(self.device), "use_cache": False}
        if keep and self.has_keep:
            kw["logits_to_keep"] = int(keep)
        with self.autocast():
            if "image_features" in batch:
                with torch.no_grad():
                    is_img = ids == self.image_token_id
                    lookup = ids.masked_fill(is_img, 0) if self.image_token_id >= self.vocab_size else ids
                    emb = self.embed(lookup)
                    feats = batch["image_features"].to(emb.device, emb.dtype)
                    if int(is_img.sum()) * emb.shape[-1] != feats.numel():
                        raise RuntimeError("image tokens and cached image features do not match")
                    emb = emb.masked_scatter(is_img.unsqueeze(-1), feats)
                logits = self.model(inputs_embeds=emb, **kw).logits
            else:
                pix = batch["pixel_values"].to(self.device, self.vision_dtype)
                logits = self.model(input_ids=ids, pixel_values=pix, **kw).logits
        if keep and logits.shape[1] != keep:
            logits = logits[:, -keep:, :]
        return logits.float()

    def per_example_ce(self, batch):
        t = trailing_supervised(batch)
        if t:
            logits = self.forward_logits(batch, keep=t + 1)
            return ce_trailing(self.torch, logits, batch["input_ids"][:, -t:].to(logits.device))
        logits = self.forward_logits(batch, keep=0)
        return ce_general(self.torch, logits, batch["labels"].to(logits.device))

    def set_train(self, peft_model):
        peft_model.train()
        self.vision_tower.eval()   # frozen; eval also keeps it out of gradient checkpointing

    def set_eval(self, peft_model):
        peft_model.eval()


# --------------------------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------------------------
def parse_args(argv=None):
    env = os.environ.get
    ap = argparse.ArgumentParser(
        description="Step-4 LoRA fine-tune of MedGemma 1.5 4B on the MHIST training pool (private Kaggle GPU). "
                    "All protocol hyperparameters are fixed in the file; the options below only say where things "
                    "are and how to fit the hardware.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Resume: run the same command again (in a new Kaggle session: attach the earlier output as an input). "
               "Exit status: 0 = finished or already finished; 75 = paused with a checkpoint; on Kaggle a pause or a "
               "failure with a checkpoint also exits 0 (see --exit-zero) and train_summary.json 'status' tells which.")
    g = ap.add_argument_group("paths")
    g.add_argument("--model-dir", default=env("MEDGEMMA_DIR") or None,
                   help="attached private dataset holding the official MedGemma 1.5 4B files [env MEDGEMMA_DIR; "
                        "if neither is given, the one dataset under /kaggle/input that holds the weights]")
    g.add_argument("--data-dir", default=env("MHIST_FT_DATA_DIR") or env("MHIST_BUNDLE_DIR") or None,
                   help="training bundle: gridded/<image>.png, train_manifest.csv, prompts/cte_p1.txt [env "
                        "MHIST_FT_DATA_DIR or MHIST_BUNDLE_DIR; else the one dataset under /kaggle/input that has them]")
    g.add_argument("--out-dir", default=env("MHIST_FT_OUT_DIR", "/kaggle/working/step4_lora"),
                   help="everything is written here [env MHIST_FT_OUT_DIR]")
    g.add_argument("--resume-from", default="auto",
                   help="where to look for an earlier session's output to continue from: a directory, 'none', or "
                        "'auto' = search /kaggle/input (attach the earlier kernel's output as an input). It is "
                        "copied into --out-dir only if it has the same configuration and is further along")
    g.add_argument("--feature-cache-dir", default=os.path.join(tempfile.gettempdir(), "mhist_step4_feature_cache"),
                   help="scratch directory for cached image features (not an output; safe to delete)")
    g = ap.add_argument_group("hardware")
    g.add_argument("--precision", default="auto", choices=["auto", "bf16", "4bit", "float32"],
                   help="auto = bf16 weights if the GPU has native bf16 (capability >= 8), else 4-bit nf4 "
                        "(bitsandbytes). The choice is recorded in train_summary.json")
    g.add_argument("--compute-dtype", default="auto", choices=["auto", "bf16", "float32"],
                   help="4-bit path only: arithmetic dtype. auto = bf16 on native-bf16 GPUs, float32 on a T4")
    g.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="cpu is for the local unit tests only")
    g.add_argument("--device-map", default="single", choices=["single", "auto", "balanced"],
                   help="single = everything on GPU 0; auto / balanced = let accelerate spread the model over all "
                        "GPUs (e.g. unquantised float32 weights over two T4s with --precision float32)")
    g.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa"],
                   help="eager is what Google's notebook uses for training")
    g.add_argument("--image-backend", default="default", choices=["default", "pil"],
                   help="default = AutoProcessor's own choice (same as the evaluation script); pil = force the "
                        "PIL image pipeline. The class used is recorded")
    g.add_argument("--micro-batch", type=int, default=2,
                   help="examples per forward pass; gradients are accumulated to the fixed effective batch of "
                        f"{EFFECTIVE_BATCH}. Halved automatically on a CUDA out-of-memory error")
    g.add_argument("--eval-batch", type=int, default=4,
                   help="validation batch size (at most 2 on the pixel path; halved automatically on a CUDA "
                        "out-of-memory error)")
    g.add_argument("--feature-cache", default="auto", choices=["auto", "on", "off"],
                   help="run the frozen vision tower once per tile and reuse its output. auto = use it if it "
                        "reproduces the pixel path on this model, else fall back to the pixel path")
    g.add_argument("--feature-batch", type=int, default=2, help="tiles per vision-tower pass when caching")
    g.add_argument("--no-gradient-checkpointing", action="store_true",
                   help="skip the recomputation pass: faster (roughly a third less compute, not measured), same "
                        "gradients, but every layer's activations are kept: about 12 GB per example at float32 "
                        "(about 365 MB x 34 layers for 831 tokens). For 24 GB GPUs only; refused with float32 "
                        "arithmetic on a GPU with less than 20 GB (a T4)")
    g = ap.add_argument_group("run control")
    g.add_argument("--time-budget-hours", type=float, default=11.0,
                   help="stop cleanly with a checkpoint after this many hours in this session (Kaggle limit: 12)")
    g.add_argument("--exit-zero", default="auto", choices=["auto", "yes", "no"],
                   help="Kaggle discards /kaggle/working when a kernel exits non-zero. With yes, a pause or a failure "
                        "that leaves a resume checkpoint still exits 0; the outcome is in train_summary.json "
                        "('status'), FAILED.txt and the log. auto = yes when /kaggle/working exists")
    g.add_argument("--save-every-steps", type=int, default=25, help="optimiser steps between resume checkpoints")
    g.add_argument("--log-every-steps", type=int, default=10, help="optimiser steps between progress lines")
    g.add_argument("--skip-baseline-eval", action="store_true",
                   help="do not score the untrained model on the validation slice (logged as epoch 0, never selectable)")
    g.add_argument("--verify-only", action="store_true",
                   help="check bundle, split, packages and tokenisation, print the plan, load no weights")
    g.add_argument("--no-hash-weights", action="store_true", help="skip the sha256 of the weight shards")
    g = ap.add_argument_group("smoke tests only (the run is then marked protocol_run=false)")
    g.add_argument("--allow-pool-mismatch", action="store_true", help="accept a manifest that is not the full pool")
    g.add_argument("--allow-model-mismatch", action="store_true", help="accept a model dir that is not the frozen revision")
    g.add_argument("--debug-limit-train", type=int, default=0, help="use only the first N training tiles")
    g.add_argument("--debug-limit-val", type=int, default=0, help="use only the first N validation tiles")
    g.add_argument("--debug-epochs", type=int, default=0, help=f"run N epochs instead of {MAX_EPOCHS}")
    g.add_argument("--debug-stop-after-steps", type=int, default=0,
                   help="pause (exit 75) after N optimiser steps in this session, to exercise resuming")
    if argv is None:
        # Arguments that are there are used, whatever argv[0] is called: kaggle_push.py's header puts them into
        # sys.argv even when a notebook kernel runs the script. Only a kernel's own `-f <connection file>` (or no
        # argument at all) falls back to the environment variable TRAIN_LORA_ARGS.
        argv = sys.argv[1:]
        if argv[:1] == ["-f"] or (not argv and os.environ.get("TRAIN_LORA_ARGS")):
            argv = shlex.split(os.environ.get("TRAIN_LORA_ARGS", ""))
    args = ap.parse_args(argv)
    if args.micro_batch < 1 or args.eval_batch < 1 or args.feature_batch < 1:
        ap.error("batch sizes must be >= 1")
    return args


def gradient_checkpointing_problem(no_gradient_checkpointing, prec):
    """Why --no-gradient-checkpointing cannot be used with this precision on this GPU (None = no objection)."""
    gpu_gb = (prec.get("gpu") or {}).get("memory_gb")
    if no_gradient_checkpointing and prec.get("compute_dtype") == "float32" and gpu_gb is not None and gpu_gb < 20:
        return (f"--no-gradient-checkpointing with float32 arithmetic keeps about 12 GB of activations per example "
                f"(about 365 MB x 34 layers); this GPU has {gpu_gb} GB. Drop the flag: the gradients are the same "
                "with checkpointing.")
    return None


def finalize_best_adapter(best_dir, ckpt_root, chosen_epoch, epochs_completed, epochs_planned):
    """Mark best_adapter/ as the FINAL adapter of a finished run: the one chosen on the validation slice after
    every planned epoch was validated. Until this has run, best_adapter/ is only the best so far, and every
    checkpoints/epoch_N/ adapter stays non-final for good. The evaluation side (infer_jobs.py, import_results.py)
    accepts dev answers only from a final adapter of a protocol run. Idempotent; returns the adapter's sha256."""
    dst = os.path.join(best_dir, "adapter_model.safetensors")
    meta_path = os.path.join(best_dir, "step4_meta.json")
    if not (os.path.isfile(dst) and os.path.isfile(meta_path)):
        die(f"{best_dir} is incomplete (adapter_model.safetensors / step4_meta.json); rerun to rebuild it")
    if epochs_completed != epochs_planned:
        die(f"only {epochs_completed} of {epochs_planned} epochs are validated: the best adapter so far is not final")
    sha = sha256_file(dst)
    src = os.path.join(ckpt_root, f"epoch_{chosen_epoch}", "adapter_model.safetensors")
    if os.path.exists(src) and sha256_file(src) != sha:
        die("best_adapter does not match the chosen epoch's adapter; rerun to rebuild it")
    meta = read_json(meta_path)
    if meta.get("epoch") != chosen_epoch:
        die(f"best_adapter/step4_meta.json is for epoch {meta.get('epoch')}, but the chosen epoch is {chosen_epoch}")
    want = {"final": True, "chosen_epoch": chosen_epoch, "epochs_completed": epochs_completed,
            "epochs_planned": epochs_planned, "adapter_model_sha256": sha}
    if any(meta.get(k) != v for k, v in want.items()):
        meta.update(want)
        atomic_write_json(meta_path, meta)
    return sha


# --------------------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    flags = {k: v for k, v in sorted(vars(args).items()) if k.startswith(("debug_", "allow_")) and v}
    log(f"arguments as parsed: verify_only={args.verify_only} model_dir={args.model_dir!r} data_dir={args.data_dir!r} "
        f"out_dir={args.out_dir!r} resume_from={args.resume_from!r} precision={args.precision} device={args.device} "
        f"debug/allow flags={flags or 'none'}" + ("" if not flags else "  -> NOT a protocol run"))
    t_session = time.time()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_root = os.path.join(out_dir, "checkpoints")
    last_dir = os.path.join(ckpt_root, "last")
    best_dir = os.path.join(out_dir, "best_adapter")
    val_dir = os.path.join(out_dir, "val_scores")
    summary_path = os.path.join(out_dir, "train_summary.json")
    epoch_log_path = os.path.join(out_dir, "train_log.jsonl")
    step_log_path = os.path.join(out_dir, "train_steps.jsonl")
    for d in (ckpt_root, val_dir):
        os.makedirs(d, exist_ok=True)
    for d in (last_dir, best_dir):
        recover_dir(d)

    # ---- 1. bundle, split, plan (no torch yet) -------------------------------------------------------------
    failed_path = os.path.join(out_dir, "FAILED.txt")
    if os.path.exists(failed_path):
        os.replace(failed_path, os.path.join(out_dir, "FAILED.previous.txt"))
    model_dir = resolve_dir(args.model_dir, ["config.json"], "--model-dir") if args.model_dir else \
        discover_dir(looks_like_model_dir, "--model-dir", "the model weights")
    data_dir = resolve_dir(args.data_dir, ["train_manifest.csv"], "--data-dir") if args.data_dir else \
        discover_dir(looks_like_bundle_dir, "--data-dir", "the training bundle")
    log(f"model dir {model_dir}")
    log(f"data dir  {data_dir}")
    log(f"out dir   {out_dir}")
    bundle = load_bundle(data_dir, allow_pool_mismatch=args.allow_pool_mismatch)
    rows = bundle["rows"]
    by_name = {r["image"]: r for r in rows}
    split = make_split(rows)
    split["pool_sha256"] = bundle["pool_sha256"]
    split["val_sha256"] = sha256_bytes("\n".join(split["val"]).encode())
    if bundle["pool_verified"] and split["val_sha256"] != EXPECTED_VAL_SHA256:
        die(f"the validation slice computed here ({split['val_sha256'][:16]}...) is not the frozen one "
            f"({EXPECTED_VAL_SHA256[:16]}...); the split code or the manifest labels / votes changed")
    split_path = os.path.join(out_dir, "split.json")
    if os.path.exists(split_path):
        old = read_json(split_path)
        if old.get("train") != split["train"] or old.get("val") != split["val"]:
            die(f"{split_path} exists and differs from the split recomputed from this bundle. "
                "Use a fresh --out-dir or the original bundle.")
    else:
        atomic_write_json(split_path, split)
    assert not set(split["train"]) & set(split["val"])
    train_names, val_names = split["train"], split["val"]
    debug = bool(args.debug_limit_train or args.debug_limit_val or args.debug_epochs or args.debug_stop_after_steps)
    if args.debug_limit_train:
        train_names = sorted(hash_order(train_names, SEED, "debug-train")[: args.debug_limit_train])
    if args.debug_limit_val:
        val_names = sorted(hash_order(val_names, SEED, "debug-val")[: args.debug_limit_val])
    train_rows = [by_name[n] for n in train_names]
    val_rows = [by_name[n] for n in val_names]
    weights, train_counts = class_weights(train_rows)
    n_epochs = args.debug_epochs or MAX_EPOCHS
    groups_per_epoch = int(math.ceil(len(train_names) / EFFECTIVE_BATCH))
    total_steps = groups_per_epoch * n_epochs
    warmup_steps = warmup_steps_for(total_steps)
    log(f"pool {len(rows)} tiles (frozen pool verified: {bundle['pool_verified']}); "
        f"train {len(train_names)} {train_counts}, validation {len(val_names)} "
        f"{ {c: sum(r['label'] == c for r in val_rows) for c in CLASSES} }")
    log(f"class weights {{{', '.join(f'{c}: {w:.4f}' for c, w in weights.items())}}}; {n_epochs} epochs x "
        f"{groups_per_epoch} optimiser steps = {total_steps} steps, warm-up {warmup_steps}, peak lr {LEARNING_RATE}")

    # ---- 2. packages, processor, tokenisation -----------------------------------------------------------------
    import torch

    prec = resolve_precision(args, torch) if not args.verify_only or (args.device == "cuda" and torch.cuda.is_available()) \
        else {"mode": "not resolved (--verify-only without a GPU)", "used_4bit": None, "autocast": False}
    versions, package_problem = package_versions(need_bnb=prec.get("mode") == "4bit",
                                                 need_accelerate=args.device == "cuda")
    if package_problem and not args.verify_only:
        die(package_problem)
    log("packages " + ", ".join(f"{k} {v}" for k, v in versions.items() if v))
    log(f"precision: {prec.get('mode')} -- {prec.get('reason', '')}")
    gc_problem = gradient_checkpointing_problem(args.no_gradient_checkpointing, prec)
    if gc_problem:
        die(gc_problem)
    from PIL import Image

    processor = load_processor(model_dir, args.image_backend)
    builder = ExampleBuilder(processor, bundle["prompt_text"])
    first = train_names[0]
    with Image.open(bundle["paths"][first]) as im:
        first_image = im.convert("RGB")
    prompt_ids, first_pixels = builder.encode_prompt(first_image)
    one_step_check = "identical"
    try:
        ref = builder.official_one_step(first_image)
        if ref["input_ids"][0].tolist() != prompt_ids.tolist():
            die("the prompt built here is not token-identical to processor.apply_chat_template(tokenize=True)")
        if not torch.equal(ref["pixel_values"][0], first_pixels):
            die("pixel_values differ from processor.apply_chat_template(tokenize=True) for the same image")
    except SystemExit:
        raise
    except Exception as e:   # an older transformers without one-step image templating: the two-step path is still official
        one_step_check = f"one-step API unavailable in this transformers ({type(e).__name__}); two-step encoding used"
    if len(prompt_ids) != EXPECTED_PROMPT_TOKENS:
        die(f"the prompt has {len(prompt_ids)} tokens, expected {EXPECTED_PROMPT_TOKENS} (PLAN amendment 2)")
    tokenization = builder.describe()
    tokenization["one_step_apply_chat_template_check"] = one_step_check
    tokenization["image_processor_class"] = type(processor.image_processor).__name__
    tokenization["tokenizer_class"] = type(processor.tokenizer).__name__
    log(f"prompt {len(prompt_ids)} tokens ({one_step_check}); targets "
        + "; ".join(f"{c}: {tokenization['target_tokens'][c]}" for c in CLASSES)
        + f"; label read by {tokenization['scoring_mode']} over {tokenization['scored_tokens']}")

    model_fp = model_fingerprint(model_dir, hash_weights=not args.no_hash_weights and not args.verify_only,
                                 allow_mismatch=args.allow_model_mismatch,
                                 cache_path=os.path.join(out_dir, "model_files.json"))
    protocol_run = bool(bundle["pool_verified"] and model_fp["matches_frozen_revision_files"] and not debug)
    hyper = {"seed": SEED, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
             "learning_rate": LEARNING_RATE, "schedule": "cosine", "warmup_fraction": WARMUP_FRACTION,
             "warmup_steps": warmup_steps, "effective_batch": EFFECTIVE_BATCH, "epochs": n_epochs,
             "optimizer_steps_per_epoch": groups_per_epoch, "total_optimizer_steps": total_steps,
             "max_grad_norm": MAX_GRAD_NORM, "optimizer": "AdamW", "betas": list(ADAM_BETAS), "eps": ADAM_EPS,
             "weight_decay": WEIGHT_DECAY, "target_template": TARGET_TEMPLATE, "target_variant": TARGET_VARIANT,
             "loss": "per-example mean cross-entropy over the target tokens, times the class weight, "
                     "averaged over the accumulation group",
             "gradient_checkpointing": not args.no_gradient_checkpointing,
             "attn_implementation": args.attn_implementation}
    # gradient checkpointing changes memory and speed, not the gradients, so it may differ between sessions
    fingerprint = canonical_sha256({
        "hyper": {k: v for k, v in hyper.items() if k != "gradient_checkpointing"}, "train": train_names, "val": val_names, "prompt": bundle["prompt_sha256"],
        "gridded": bundle["gridded_sha256"], "model": model_fp["small_files_sha256"],
        "precision": {k: prec.get(k) for k in ("mode", "compute_dtype", "weights_dtype", "quantized_scope")},
        "image_processor": tokenization["image_processor_class"], "targets": builder.target_ids})

    if args.verify_only:
        print(json.dumps({"split_sizes": {"train": len(train_names), "val": len(val_names)},
                          "strata": split["strata"], "class_weights": weights, "plan": hyper,
                          "tokenization": tokenization, "precision": prec, "package_versions": versions,
                          "protocol_run": protocol_run}, indent=2))
        if package_problem:
            log("verify-only: bundle, split and tokenisation are fine, but: " + package_problem)
            return 2
        log("verify-only: all checks passed; no weights were loaded and nothing was trained.")
        return 0

    # ---- 3. resume state --------------------------------------------------------------------------------------
    state = None
    state_path = os.path.join(last_dir, "state.json")
    if os.path.exists(state_path) and read_json(state_path).get("fingerprint") != fingerprint:
        die(f"{last_dir} belongs to a different configuration (data, prompt, model, precision or settings "
            "changed). Use a fresh --out-dir; nothing was overwritten.")
    if args.resume_from != "none":
        explicit = args.resume_from != "auto"
        found = find_resumable(args.resume_from if explicit else KAGGLE_INPUT, fingerprint, not_under=out_dir)
        if explicit and found is None:
            die(f"--resume-from {args.resume_from}: no checkpoints/last with this run's configuration in there")
        local = run_progress(read_json(state_path)) if os.path.exists(state_path) else (-1, -1)
        if found is not None and found[0] > local:
            log(f"continuing the earlier session found in {found[1]} (optimiser step {found[0][0]}, "
                f"{found[0][1]} epoch(s) validated)")
            for item in RUN_ITEMS:
                src, dst = os.path.join(found[1], item), os.path.join(out_dir, item)
                if not os.path.exists(src):
                    continue
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                copy_tree_writable(src, dst)
    if os.path.exists(state_path):
        state = read_json(state_path)
    if os.path.exists(summary_path):
        old = read_json(summary_path)
        if old.get("status") == "complete" and old.get("fingerprint") == fingerprint \
                and os.path.isdir(best_dir) and (state or {}).get("epoch", 0) >= n_epochs:
            finalize_best_adapter(best_dir, ckpt_root, old.get("chosen_epoch"), len(state["history"]), n_epochs)
            log(f"already complete: best epoch {old.get('chosen_epoch')} "
                f"(validation balanced accuracy {old.get('chosen_validation', {}).get('balanced_accuracy')}). "
                "Nothing to do.")
            return 0
    if state is None:
        state = {"fingerprint": fingerprint, "epoch": 0, "next_group": 0, "global_step": 0,
                 "micro_batch": args.micro_batch,
                 "epoch_stats": {"sum_w_ce": 0.0, "sum_ce": 0.0, "sum_w": 0.0, "n": 0, "sum_grad_norm": 0.0,
                                 "steps": 0, "seconds": 0.0},
                 "history": [], "baseline": None, "best": None,
                 "seconds": {"wall": 0.0, "train": 0.0, "validation": 0.0, "features": 0.0, "model_load": 0.0},
                 "sessions": 0, "events": []}
        resumed = False
    else:
        resumed = True
        log(f"resuming: epoch {state['epoch'] + 1}, optimiser step {state['global_step']} of {total_steps}, "
            f"{len(state['history'])} epoch(s) validated")
    state["sessions"] += 1
    wall_prior = state["seconds"]["wall"]

    # ---- 4. model + LoRA --------------------------------------------------------------------------------------
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.manual_seed(SEED)
    t0 = time.time()
    log("loading the model ...")
    model = load_model(args, model_dir, prec)
    targets, other_linear = select_lora_targets(model)
    check_lora_targets(model, targets)
    use_gc = not args.no_gradient_checkpointing
    gc_kwargs = {"use_reentrant": False}
    if prec["mode"] == "4bit":
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gc,
                                                gradient_checkpointing_kwargs=gc_kwargs)
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        if use_gc:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
    peft_model = attach_lora(model, targets, torch)
    lora_info = verify_lora(peft_model, targets)
    lora_info["not_adapted_linear_layers"] = {
        "vision_tower": sum("vision_tower" in n.split(".") for n in other_linear),
        "other": sorted(n for n in other_linear if "vision_tower" not in n.split("."))}
    engine = Engine(torch, model, prec, builder.image_token_id)
    if any(p.requires_grad for p in engine.vision_tower.parameters()):
        die("a vision-tower parameter is trainable")
    config_image_id = getattr(model.config, "image_token_id", None)
    if config_image_id is None:
        config_image_id = getattr(model.config, "image_token_index", None)
    if config_image_id != builder.image_token_id:
        die(f"model config image token id {config_image_id} != tokenizer <image_soft_token> id {builder.image_token_id}")
    quantized = sum(type(m).__name__ == "Linear4bit" for m in model.modules())
    prec["n_linear4bit_modules"] = quantized
    prec["embedding_dtype"] = str(engine.embed_dtype).replace("torch.", "")
    state["seconds"]["model_load"] += time.time() - t0
    log(f"model ready in {time.time() - t0:.0f}s: LoRA on {lora_info['n_target_modules']} language-model linear layers, "
        f"{lora_info['trainable_parameters']:,} trainable parameters; vision tower frozen "
        f"({lora_info['not_adapted_linear_layers']['vision_tower']} linear layers not adapted); "
        f"{quantized} 4-bit layers; logits_to_keep supported: {engine.has_keep}")

    named_trainable = sorted(((n, p) for n, p in peft_model.named_parameters() if p.requires_grad), key=lambda x: x[0])
    trainable = [p for _, p in named_trainable]
    opt_kw = {"lr": LEARNING_RATE, "betas": ADAM_BETAS, "eps": ADAM_EPS, "weight_decay": WEIGHT_DECAY}
    try:
        optimizer = torch.optim.AdamW(trainable, fused=(args.device == "cuda"), **opt_kw)
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(trainable, **opt_kw)
    if resumed:
        blob = torch.load(os.path.join(last_dir, "trainable.pt"), map_location="cpu")
        if sorted(blob) != [n for n, _ in named_trainable]:
            die(f"{last_dir}/trainable.pt does not hold the same tensors as this model")
        with torch.no_grad():
            for n, p in named_trainable:
                p.copy_(blob[n].to(p.device, p.dtype))
        optimizer.load_state_dict(torch.load(os.path.join(last_dir, "optimizer.pt"), map_location="cpu"))
        del blob

    def save_last():
        tmp = last_dir + ".tmp"
        if os.path.isdir(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)
        torch.save({n: p.detach().cpu() for n, p in named_trainable}, os.path.join(tmp, "trainable.pt"))
        torch.save(optimizer.state_dict(), os.path.join(tmp, "optimizer.pt"))
        state["seconds"]["wall"] = wall_prior + (time.time() - t_session)
        atomic_write_json(os.path.join(tmp, "state.json"), state)
        swap_dir(tmp, last_dir)

    def save_adapter(final_dir, meta):
        recover_dir(final_dir)
        tmp = final_dir + ".tmp"
        # LoRA matrices only: never the 2.7 GB embedding matrix, and no look-up of the base model anywhere
        peft_model.save_pretrained(tmp, save_embedding_layers=False)
        atomic_write_json(os.path.join(tmp, "step4_meta.json"), meta)
        swap_dir(tmp, final_dir)

    # ---- 5. image features ------------------------------------------------------------------------------------
    def open_tile(name):
        with Image.open(bundle["paths"][name]) as im:
            return im.convert("RGB")

    feature_index, feature_bank, feature_info = None, None, {"mode": "pixel path (vision tower run every step)"}
    needed = sorted(set(train_names) | set(val_names))
    if args.feature_cache != "off":
        engine.set_eval(peft_model)
        probe_names = hash_order(train_names, SEED, "feature-probe")[:2]
        worst, probe_error = 0.0, None
        try:
            for name in probe_names:
                pix = builder.encode_image(open_tile(name)).unsqueeze(0)
                ids = builder.scoring_ids()
                base = {k: v.unsqueeze(0) for k, v in ids.items()}
                with torch.no_grad():
                    a = engine.forward_logits(dict(base, pixel_values=pix), keep=1)[:, -1, :]
                    b = engine.forward_logits(dict(base, image_features=engine.image_features(pix)), keep=1)[:, -1, :]
                worst = max(worst, float((a - b).abs().max() / max(1.0, float(a.abs().max()))))
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:   # this transformers version does not take the cached route: use the pixel path
            worst, probe_error = float("inf"), f"{type(e).__name__}: {str(e)[:200]}"
        tol = 2e-2 if prec.get("compute_dtype") == "bfloat16" else 1e-3
        feature_info = {"probe_tiles": probe_names, "max_relative_logit_difference": worst if probe_error is None else None,
                        "tolerance": tol, "probe_error": probe_error}
        if worst <= tol:
            feature_info["mode"] = "cached image features (frozen vision tower + projector run once per tile)"
        elif args.feature_cache == "on":
            die(f"--feature-cache on, but the cached path differs from the pixel path by {worst:.3g} (> {tol})"
                + (f"; {probe_error}" if probe_error else ""))
        else:
            feature_info["mode"] = "pixel path (cached path did not reproduce the pixel path; fell back)"
            log(f"feature cache disabled: relative logit difference {worst:.3g} > {tol}"
                + (f"; {probe_error}" if probe_error else ""))
    use_cache = feature_info["mode"].startswith("cached")
    if use_cache:
        t0 = time.time()
        key = canonical_sha256({"fp": fingerprint, "names": needed})
        os.makedirs(args.feature_cache_dir, exist_ok=True)
        cache_file = os.path.join(args.feature_cache_dir, f"features_{key[:20]}.pt")
        if os.path.exists(cache_file):
            try:
                blob = torch.load(cache_file, map_location="cpu")
                if blob.get("key") == key and blob.get("names") == needed:
                    feature_bank = blob["features"]
                    log(f"image features loaded from {cache_file}")
            except Exception as e:
                log(f"ignoring unreadable feature cache ({type(e).__name__})")
        if feature_bank is None:
            fb = args.feature_batch
            i = 0
            while i < len(needed):
                chunk = needed[i:i + fb]
                pix = torch.stack([builder.encode_image(open_tile(n)) for n in chunk])
                try:
                    feats = engine.image_features(pix).cpu()
                except torch.cuda.OutOfMemoryError:
                    if fb == 1:
                        raise
                    fb = max(1, fb // 2)
                    torch.cuda.empty_cache()
                    log(f"out of memory in the vision tower; feature batch -> {fb}")
                    continue
                if feature_bank is None:
                    feature_bank = torch.empty((len(needed),) + tuple(feats.shape[1:]), dtype=feats.dtype)
                feature_bank[i:i + len(chunk)] = feats
                i += len(chunk)
                if i == len(chunk) or i % (50 * fb) < fb or i == len(needed):
                    rate = (time.time() - t0) / i
                    log(f"image features {i}/{len(needed)}  {rate:.2f}s/tile  ETA {rate * (len(needed) - i) / 60:.1f} min")
            try:
                tmp = cache_file + ".tmp"
                torch.save({"key": key, "names": needed, "features": feature_bank}, tmp)
                os.replace(tmp, cache_file)
            except OSError as e:   # scratch disk full: the features stay in memory for this session
                log(f"could not store the feature cache on disk ({e}); continuing with the in-memory copy")
        feature_index = {n: i for i, n in enumerate(needed)}
        feature_info.update({"n_tiles": len(needed), "shape_per_tile": list(feature_bank.shape[1:]),
                             "dtype": str(feature_bank.dtype).replace("torch.", "")})
        state["seconds"]["features"] += time.time() - t0
    log(f"image input: {feature_info['mode']}")

    def image_inputs(names):
        if use_cache:
            return {"image_features": feature_bank[[feature_index[n] for n in names]]}
        return {"pixel_values": torch.stack([builder.encode_image(open_tile(n)) for n in names])}

    def train_batch(names):
        batch = collate([builder.training_ids(by_name[n]["label"]) for n in names], builder.pad_id, torch)
        batch.update(image_inputs(names))
        batch["weights"] = torch.tensor([weights[by_name[n]["label"]] for n in names], dtype=torch.float32)
        return batch

    # ---- 6. validation ----------------------------------------------------------------------------------------
    scored = builder.scoring["scored_ids"]
    contexts = sorted({tuple(v[:-1]) for v in scored.values()})

    # On the pixel path every validation batch also goes through the vision tower with eager attention
    # (16 heads x 4096 x 4096 per tile at float32, about 1 GB per attention tensor): 2 tiles at most.
    eval_state = {"batch": args.eval_batch if use_cache else min(args.eval_batch, 2)}

    def validate(tag):
        """Score every validation tile: log P(label tokens | prompt + shared target prefix), HP vs SSA.
        A CUDA out-of-memory error halves the batch and redoes it (the scores do not depend on the batch size
        beyond float level), as the training step does; only at batch size 1 is it fatal."""
        engine.set_eval(peft_model)
        t0 = time.time()
        records = []
        i = 0
        while i < len(val_names):
            names = val_names[i:i + eval_state["batch"]]
            scores = {c: None for c in CLASSES}
            images = batch = logp = None
            try:
                images = image_inputs(names)
                for ctx in contexts:
                    batch = collate([builder.scoring_ids(ctx)] * len(names), builder.pad_id, torch)
                    batch.update(images)
                    with torch.no_grad():
                        logp = torch.log_softmax(engine.forward_logits(batch, keep=len(ctx) + 1), dim=-1)
                    for c in CLASSES:
                        seq = scored[c]
                        if tuple(seq[:-1]) == ctx:
                            scores[c] = sum(logp[:, j, tok] for j, tok in enumerate(seq)).cpu().tolist()
            except torch.cuda.OutOfMemoryError:
                images = batch = logp = None
                torch.cuda.empty_cache()
                if eval_state["batch"] == 1:
                    die(f"CUDA out of memory in validation ({tag}) at batch size 1. Free the GPU, or use --precision 4bit.")
                old_batch, eval_state["batch"] = eval_state["batch"], max(1, eval_state["batch"] // 2)
                state["events"].append({"step": state["global_step"],
                                        "event": f"out of memory in validation ({tag}), eval batch {old_batch} -> {eval_state['batch']}"})
                log(f"CUDA out of memory in validation ({tag}): eval batch {old_batch} -> {eval_state['batch']}, redoing the batch")
                continue
            i += len(names)
            for k, n in enumerate(names):
                s_hp, s_ssa = float(scores["HP"][k]), float(scores["SSA"][k])
                if not (math.isfinite(s_hp) and math.isfinite(s_ssa)):
                    die(f"non-finite validation score for {n} ({tag})")
                records.append({"image": n, "label": by_name[n]["label"], "score_HP": s_hp, "score_SSA": s_ssa,
                                "pred": "SSA" if s_ssa > s_hp else "HP"})
        metrics = val_metrics(records)
        metrics["seconds"] = round(time.time() - t0, 1)
        state["seconds"]["validation"] += time.time() - t0
        return metrics, records

    def write_summary(status):
        state["seconds"]["wall"] = wall_prior + (time.time() - t_session)
        best = state["best"]
        chosen = next((h for h in state["history"] if best and h["epoch"] == best["epoch"]), None)
        best_sha = None
        best_file = os.path.join(best_dir, "adapter_model.safetensors")
        if os.path.exists(best_file):
            best_sha = sha256_file(best_file)
        summary = {
            "status": status,
            "protocol": "runs/competence/PLAN.md, 'Addendum: step 4, LoRA fine-tune on private cloud GPU'",
            "protocol_run": protocol_run,
            "debug_flags": {k: v for k, v in vars(args).items() if k.startswith(("debug_", "allow_")) and v},
            "fingerprint": fingerprint,
            "chosen_epoch": best["epoch"] if best else None,
            "chosen_validation": chosen["val"] if chosen else None,
            "selection_rule": "highest validation balanced accuracy (label read from the HP-vs-SSA logits at the "
                              "label position); ties go to the earlier epoch",
            "epochs_completed": len(state["history"]), "epochs_planned": n_epochs,
            "validation_per_epoch": [{"epoch": h["epoch"], **h["val"]} for h in state["history"]],
            "train_per_epoch": [{"epoch": h["epoch"], **h["train"]} for h in state["history"]],
            "baseline_untrained_epoch0": state["baseline"],
            "class_weights": weights, "class_counts_train": train_counts,
            "class_counts_val": {c: sum(r["label"] == c for r in val_rows) for c in CLASSES},
            "split": {"n_pool": len(rows), "n_train": len(train_names), "n_val": len(val_names),
                      "strata": split["strata"], "file": "split.json", "pool_verified": bundle["pool_verified"]},
            "precision": prec, "used_4bit": prec["used_4bit"],
            "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "bias": "none",
                     "task_type": "CAUSAL_LM", "target": "every nn.Linear of the language model "
                     f"({', '.join(LM_LINEAR_LEAVES)} in each block); vision tower, projector and lm_head untouched",
                     **lora_info},
            "hyperparameters": hyper, "micro_batch_final": state["micro_batch"], "eval_batch_final": eval_state["batch"],
            "choices_not_fixed_by_the_addendum": {
                "max_grad_norm": MAX_GRAD_NORM, "adamw_betas": list(ADAM_BETAS), "adamw_eps": ADAM_EPS,
                "weight_decay": WEIGHT_DECAY,
                "loss": "mean cross-entropy over the target tokens of an example, times its class weight, averaged "
                        "over the 8 examples of an optimiser step",
                "lora_targets": "the 7 linear layers of each language-model block; lm_head is not adapted",
                "ties": "equal validation balanced accuracy -> the earlier epoch",
                "4bit_scope": prec.get("quantized_scope") or "not used",
                "target_variant": TARGET_VARIANT},
            "tokenization": tokenization, "image_input": feature_info,
            "model": model_fp,
            "data": {k: bundle[k] for k in ("prompt_sha256", "pool_sha256", "manifest_sha256", "gridded_sha256",
                                            "n_unlisted_files_in_gridded")},
            "package_versions": versions,
            "wall_time_seconds": {k: round(v, 1) for k, v in state["seconds"].items()},
            "sessions": state["sessions"], "events": state["events"],
            "best_adapter": {"dir": "best_adapter", "adapter_model_sha256": best_sha, "final": status == "complete"},
            "google_notebook": "github.com/google-health/medgemma notebooks/fine_tune_with_hugging_face.ipynb "
                               "(conventions adopted and changed are listed in this script's docstring)",
            "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        atomic_write_json(summary_path, summary)
        return summary

    def meta_for(epoch, val):
        return {"epoch": epoch, "validation": val, "fingerprint": fingerprint, "precision": prec,
                "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION}, "target_ids": builder.target_ids,
                "scoring": {"mode": builder.scoring["mode"], "scored_ids": scored},
                "image_processor_class": tokenization["image_processor_class"], "protocol_run": protocol_run,
                # set to true by finalize_best_adapter() in best_adapter/ only, once every epoch is validated
                "final": False, "epochs_planned": n_epochs, "target_variant": TARGET_VARIANT}

    if state["baseline"] is None and not args.skip_baseline_eval and state["global_step"] == 0:
        metrics, records = validate("baseline")
        state["baseline"] = metrics
        atomic_write_json(os.path.join(val_dir, "epoch_0_baseline.json"), {"epoch": 0, "metrics": metrics, "tiles": records})
        append_jsonl_once(epoch_log_path, {"event": "baseline", "epoch": 0, "selectable": False, "val": metrics,
                                           "time": time.strftime("%Y-%m-%d %H:%M:%S")}, ("event", "epoch"))
        log(f"untrained baseline on validation: balanced accuracy {metrics['balanced_accuracy']:.4f} "
            f"(HP {metrics['recall_HP']:.3f}, SSA {metrics['recall_SSA']:.3f}), {metrics['seconds']:.0f}s; "
            f"eval batch {eval_state['batch']}")

    # ---- 7. training ------------------------------------------------------------------------------------------
    def pause(reason):
        save_last()
        write_summary("paused")
        log(f"PAUSED ({reason}) at epoch {state['epoch'] + 1}, optimiser step {state['global_step']}/{total_steps}. "
            "Run the same command again to resume.")
        return EXIT_PAUSED

    steps_this_session = 0
    examples_this_session, train_seconds_this_session = 0, 0.0
    window = {"sum_w_ce": 0.0, "n": 0, "sum_gn": 0.0, "steps": 0}
    val_seconds_guess = (state["baseline"] or {}).get("seconds") or 0.0
    while state["epoch"] < n_epochs:
        epoch = state["epoch"]
        groups = epoch_groups(train_names, SEED, epoch)
        assert len(groups) == groups_per_epoch
        while state["next_group"] < len(groups):
            g = state["next_group"]
            names = groups[g]
            step = epoch * groups_per_epoch + g
            assert step == state["global_step"], (step, state["global_step"])
            lr = lr_at(step, total_steps, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            t_step = time.time()
            while True:
                mb = state["micro_batch"]
                torch.manual_seed(SEED * 100003 + step)   # dropout masks are a function of the step, not of history
                engine.set_train(peft_model)
                optimizer.zero_grad(set_to_none=True)
                stats = {"sum_w_ce": 0.0, "sum_ce": 0.0, "sum_w": 0.0}
                batch = ce = loss = None
                try:
                    for i in range(0, len(names), mb):
                        chunk = names[i:i + mb]
                        batch = train_batch(chunk)
                        ce = engine.per_example_ce(batch)
                        w = batch["weights"].to(ce.device)
                        loss = (w * ce).sum() / len(names)
                        if not bool(torch.isfinite(loss)):
                            save_last()
                            die(f"non-finite loss at epoch {epoch + 1}, step {step} (tiles {chunk}); the last "
                                "good state is saved in checkpoints/last")
                        loss.backward()
                        stats["sum_w_ce"] += float((w * ce.detach()).sum())
                        stats["sum_ce"] += float(ce.detach().sum())
                        stats["sum_w"] += float(w.sum())
                    break
                except torch.cuda.OutOfMemoryError:
                    batch = ce = loss = None
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    if mb == 1:
                        die("CUDA out of memory with micro-batch 1. Free the GPU, or use --precision 4bit.")
                    state["micro_batch"] = max(1, mb // 2)
                    state["events"].append({"step": step, "event": f"out of memory, micro-batch {mb} -> {state['micro_batch']}"})
                    log(f"CUDA out of memory at step {step}: micro-batch {mb} -> {state['micro_batch']}, redoing the step")
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM))
            if not math.isfinite(grad_norm):
                save_last()
                die(f"non-finite gradient norm at step {step}; the last good state is saved in checkpoints/last")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            dt = time.time() - t_step
            es = state["epoch_stats"]
            for k in ("sum_w_ce", "sum_ce", "sum_w"):
                es[k] += stats[k]
            es["n"] += len(names)
            es["sum_grad_norm"] += grad_norm
            es["steps"] += 1
            es["seconds"] += dt
            state["seconds"]["train"] += dt
            state["next_group"] = g + 1
            state["global_step"] = step + 1
            steps_this_session += 1
            examples_this_session += len(names)
            train_seconds_this_session += dt
            window["sum_w_ce"] += stats["sum_w_ce"]
            window["n"] += len(names)
            window["sum_gn"] += grad_norm
            window["steps"] += 1
            done = state["global_step"]
            if done % args.log_every_steps == 0 or done == total_steps or steps_this_session in (1, 5):
                sec_ex = train_seconds_this_session / examples_this_session
                left_examples = len(train_names) * n_epochs - (epoch * len(train_names) + es["n"])
                left_vals = n_epochs - len(state["history"])
                eta_h = (left_examples * sec_ex + left_vals * val_seconds_guess) / 3600
                rec = {"event": "step", "epoch": epoch + 1, "step_in_epoch": g + 1, "global_step": done,
                       "lr": lr, "loss_weighted": round(window["sum_w_ce"] / max(window["n"], 1), 6),
                       "grad_norm": round(window["sum_gn"] / max(window["steps"], 1), 5),
                       "sec_per_example": round(sec_ex, 3), "eta_hours": round(eta_h, 2),
                       "micro_batch": state["micro_batch"],
                       "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if args.device == "cuda" else None}
                with open(step_log_path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                log(f"epoch {epoch + 1}/{n_epochs} step {g + 1}/{len(groups)} (global {done}/{total_steps}) "
                    f"loss {rec['loss_weighted']:.4f} lr {lr:.2e} grad {rec['grad_norm']:.3f} "
                    f"{sec_ex:.2f}s/example ETA {eta_h:.2f} h"
                    + (f" GPU {rec['gpu_mem_gb']} GB" if rec["gpu_mem_gb"] is not None else ""))
                window = {"sum_w_ce": 0.0, "n": 0, "sum_gn": 0.0, "steps": 0}
            end_of_epoch = state["next_group"] == len(groups)
            if not end_of_epoch:
                if done % args.save_every_steps == 0:
                    save_last()
                if args.debug_stop_after_steps and steps_this_session >= args.debug_stop_after_steps:
                    return pause("--debug-stop-after-steps")
                if time.time() - t_session > args.time_budget_hours * 3600:
                    return pause(f"time budget of {args.time_budget_hours} h reached")
        # ---- end of epoch: checkpoint, adapter, validation, log, best ---------------------------------------
        save_last()   # training of this epoch is safe before the slower steps below
        es = state["epoch_stats"]
        epoch_no = epoch + 1
        train_stats = {"mean_loss_weighted": round(es["sum_w_ce"] / max(es["sum_w"], 1e-12), 6),
                       "mean_loss_unweighted": round(es["sum_ce"] / max(es["n"], 1), 6),
                       "n_examples": es["n"], "optimizer_steps": es["steps"],
                       "mean_grad_norm_before_clipping": round(es["sum_grad_norm"] / max(es["steps"], 1), 5),
                       "lr_last_step": lr_at(epoch_no * groups_per_epoch - 1, total_steps, warmup_steps),
                       "seconds": round(es["seconds"], 1)}
        metrics, records = validate(f"epoch {epoch_no}")
        val_seconds_guess = metrics["seconds"]
        epoch_dir = os.path.join(ckpt_root, f"epoch_{epoch_no}")
        save_adapter(epoch_dir, meta_for(epoch_no, metrics))
        atomic_write_json(os.path.join(val_dir, f"epoch_{epoch_no}.json"),
                          {"epoch": epoch_no, "metrics": metrics, "tiles": records})
        best = state["best"]
        is_best = best is None or metrics["balanced_accuracy"] > best["balanced_accuracy"]
        if is_best:
            recover_dir(best_dir)
            tmp = best_dir + ".tmp"
            shutil.copytree(epoch_dir, tmp)
            swap_dir(tmp, best_dir)
            state["best"] = {"epoch": epoch_no, "balanced_accuracy": metrics["balanced_accuracy"]}
        state["history"].append({"epoch": epoch_no, "train": train_stats, "val": metrics})
        append_jsonl_once(epoch_log_path, {
            "event": "epoch", "epoch": epoch_no, "train": train_stats, "val": metrics, "is_best_so_far": is_best,
            "best_epoch_so_far": state["best"]["epoch"], "micro_batch": state["micro_batch"],
            "precision": prec["mode"], "time": time.strftime("%Y-%m-%d %H:%M:%S")}, ("event", "epoch"))
        log(f"EPOCH {epoch_no}/{n_epochs}: train loss {train_stats['mean_loss_weighted']:.4f}; validation balanced "
            f"accuracy {metrics['balanced_accuracy']:.4f} (HP {metrics['recall_HP']:.3f}, SSA {metrics['recall_SSA']:.3f}, "
            f"accuracy {metrics['accuracy']:.3f}); best so far: epoch {state['best']['epoch']} "
            f"({state['best']['balanced_accuracy']:.4f}). Report only, not used for selection: expected under tier-1 "
            f"sampling, accuracy {metrics['expected_tier1_accuracy']:.3f} / balanced {metrics['expected_tier1_balanced_accuracy']:.3f}")
        state["epoch"] = epoch_no
        state["next_group"] = 0
        state["epoch_stats"] = {"sum_w_ce": 0.0, "sum_ce": 0.0, "sum_w": 0.0, "n": 0, "sum_grad_norm": 0.0,
                                "steps": 0, "seconds": 0.0}
        save_last()
        write_summary("in_progress")   # "complete" is written in step 8 only, after best_adapter is marked final
        if state["epoch"] < n_epochs:
            if args.debug_stop_after_steps and steps_this_session >= args.debug_stop_after_steps:
                return pause("--debug-stop-after-steps")
            if time.time() - t_session > args.time_budget_hours * 3600:
                return pause(f"time budget of {args.time_budget_hours} h reached")

    # ---- 8. done ------------------------------------------------------------------------------------------------
    # Every planned epoch is validated: best_adapter/ becomes the final, validation-selected adapter. Only now
    # can the summary say "complete", so "complete" always implies a final adapter with a recorded sha256.
    best = state["best"]
    finalize_best_adapter(best_dir, ckpt_root, best["epoch"], len(state["history"]), n_epochs)
    summary = write_summary("complete")
    log(f"DONE: chosen epoch {summary['chosen_epoch']} with validation balanced accuracy "
        f"{summary['chosen_validation']['balanced_accuracy']:.4f}; adapter in {best_dir}; "
        f"wall time {summary['wall_time_seconds']['wall'] / 3600:.2f} h over {state['sessions']} session(s)")
    return 0


def run(argv=None):
    """Command-line entry point. main() does the work; this decides the exit status.

    A Kaggle kernel that exits non-zero loses /kaggle/working, and with it the resume checkpoint. So with
    --exit-zero yes (auto on Kaggle), a pause, or a failure that leaves a checkpoint behind, still exits 0. The
    real outcome is never hidden: train_summary.json has status paused / failed, FAILED.txt holds the traceback,
    and the last log line says so."""
    args = parse_args(argv)
    keep = args.exit_zero == "yes" or (args.exit_zero == "auto" and os.path.isdir(KAGGLE_WORKING))
    out_dir = os.path.abspath(args.out_dir)
    try:
        code = main(argv)
    except BaseException as e:
        if isinstance(e, SystemExit) and e.code in (0, None):
            return 0
        has_checkpoint = os.path.exists(os.path.join(out_dir, "checkpoints", "last", "state.json"))
        message = f"{type(e).__name__}: {e}"
        try:
            os.makedirs(out_dir, exist_ok=True)
            atomic_write_text(os.path.join(out_dir, "FAILED.txt"),
                              f"{time.strftime('%Y-%m-%d %H:%M:%S')}  train_lora.py stopped with an error.\n{message}\n\n"
                              f"{traceback.format_exc()}\nresume checkpoint present: {has_checkpoint}\n")
            summary_path = os.path.join(out_dir, "train_summary.json")
            if os.path.exists(summary_path):
                summary = read_json(summary_path)
                if summary.get("status") != "complete":
                    summary.update({"status": "failed", "error": message})
                    atomic_write_json(summary_path, summary)
        except OSError:
            pass
        if keep and has_checkpoint and not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
            log(f"FAILED ({message}). Exit status 0 on purpose, so that the output and its resume checkpoint are "
                "kept; see FAILED.txt. Fix the cause and run again to resume.")
            return 0
        raise
    if code == EXIT_PAUSED and keep:
        log("NOT FINISHED: paused with a checkpoint. Exit status 0 on purpose, so that the output is kept; "
            "attach it to the next session and run again.")
        return 0
    return code


if __name__ == "__main__":
    _exit_code = run()
    if _exit_code:          # a plain return on success: inside a notebook kernel sys.exit(0) is reported as an error
        sys.exit(_exit_code)
