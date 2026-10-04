#!/usr/bin/env python3
"""CPU-only local tests for kaggle_ft/train_lora.py. They never load the MedGemma weights.

    ~/mhist_local/venv/bin/python kaggle_ft/test_train_data_local.py        (add -v for test names)

What is used from the real model directory: the processor / tokenizer files and config.json only.
What is checked:
  (a) the prompt part of a training example is token-identical to processor.apply_chat_template for the same
      image + text (and so are the pixel values);
  (b) labels are -100 everywhere except the target tokens;
  (c) the validation split is deterministic, stratified by label x agreement band, disjoint from training, and
      built from pool tiles only (frozen pool hash, no dev or test tile);
  (d) how 'HP' / 'SSA' tokenise after the target prefix, and that the label read-out follows from it;
  plus: the LoRA target list against the Gemma 3 module tree (no vision_tower module wrapped), the learning-rate
  schedule against transformers' own, the loss alignment and the cached-image-feature path on a tiny randomly
  initialised Gemma 3 (same architecture, toy widths), and an end-to-end run of train_lora.main on that tiny
  model: uninterrupted vs paused-and-resumed vs crashed-during-validation vs continued in a second "Kaggle
  session" (fresh working directory, inputs and earlier output discovered under the input directory) must all
  give the same adapter.

peft is not installed in the local venv, so the model-level tests install a minimal stand-in that wraps the
target nn.Linear modules with LoRA matrices under the same names PEFT uses. It exercises this repository's
code (targets, loop, resume, files), not PEFT itself. If real peft is importable it is used instead.

Temporary files go to a folder of this run's own under ~/mhist_local/tmp/ (MHIST_TEST_TMP names another base)
and are removed at the end; nothing is written inside the project tree, which is copied to a public repository
(the folder holds a tiny model and links to the model's tokenizer files). Two runs at once do not disturb each
other. The gridded tiles used are training-pool tiles only, rendered with run_experiment.b64_gridded_tile.
"""

import base64
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import types
import unittest

sys.dont_write_bytecode = True                  # no __pycache__ next to the project's modules
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
MODEL_DIR = os.environ.get("MEDGEMMA_DIR") or os.path.expanduser("~/mhist_local/medgemma-1.5-4b-it")
TMP_BASE = os.path.abspath(os.path.expanduser(os.environ.get("MHIST_TEST_TMP") or "~/mhist_local/tmp"))

import train_lora as tl   # noqa: E402

FINDINGS = {}
_CACHE = {}
TMP = None
N_TARGET = {"plan": 9, "label_end": 8}          # target tokens per variant (train_lora.TARGET_VARIANT says which is in force)


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def cached(fn):
    def wrapper():
        if fn.__name__ not in _CACHE:
            _CACHE[fn.__name__] = fn()
        return _CACHE[fn.__name__]
    return wrapper


@cached
def torch_mod():
    import torch
    torch.set_num_threads(4)
    return torch


@cached
def project_tables():
    """Pool rows with labels (pool tiles only) and the NAMES of dev / test tiles (no labels kept)."""
    splits = read_json(os.path.join(ROOT, "runs", "competence", "splits.json"))
    pool = set(splits["fewshot_pool"])
    rows, test_names = [], set()
    with open(os.path.join(ROOT, "..", "annotations.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            if r["Partition"] == "test":
                test_names.add(r["Image Name"])
            if r["Image Name"] in pool:
                rows.append({"image": r["Image Name"], "label": r["Majority Vote Label"],
                             "ssa_votes": int(r["Number of Annotators who Selected SSA (Out of 7)"])})
    rows.sort(key=lambda r: r["image"])
    return {"pool_rows": rows, "dev": set(splits["dev"]), "test": test_names}


def gridded_png_bytes(name):
    import run_experiment as rx
    return base64.b64decode(rx.b64_gridded_tile({"image": name, "gridded": None}))


@cached
def pool_tile():
    """One real gridded POOL tile as a PIL image."""
    from PIL import Image
    t = project_tables()
    name = t["pool_rows"][0]["image"]
    assert name not in t["dev"] and name not in t["test"]
    return name, Image.open(io.BytesIO(gridded_png_bytes(name))).convert("RGB")


@cached
def prompt_text():
    with open(os.path.join(ROOT, "prompts", "rendered", "cte_p1.txt")) as fh:
        return fh.read()


@cached
def processor():
    return tl.load_processor(MODEL_DIR, "pil")


@cached
def builder():
    b = tl.ExampleBuilder(processor(), prompt_text())
    b.encode_prompt(pool_tile()[1])
    return b


def tmp_dir():
    global TMP
    if TMP is None:
        real = os.path.realpath(TMP_BASE)
        if real.startswith(os.path.realpath(ROOT) + os.sep) or real.startswith(os.path.realpath(os.path.expanduser("~/Documents")) + os.sep):
            raise SystemExit(f"the test folder base {TMP_BASE} is inside the project tree or Documents; set MHIST_TEST_TMP elsewhere")
        os.makedirs(TMP_BASE, exist_ok=True)
        TMP = tempfile.mkdtemp(prefix="mhist_test_train_", dir=TMP_BASE)
    return TMP


def tearDownModule():
    if TMP is not None:
        shutil.rmtree(TMP, ignore_errors=True)        # this run's own folder only
    print("\n================ FINDINGS ================")
    print(json.dumps(FINDINGS, indent=2, ensure_ascii=False))


# ----------------------------------------------------------------------------------------------------------------
# peft stand-in (only when real peft is missing)
# ----------------------------------------------------------------------------------------------------------------
def _build_peft_stub():
    torch = torch_mod()
    nn = torch.nn

    class LoraConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class LoraLinear(nn.Module):
        def __init__(self, base, r, alpha, dropout):
            super().__init__()
            self.base_layer = base
            self.lora_dropout = nn.ModuleDict({"default": nn.Dropout(dropout)})
            self.lora_A = nn.ModuleDict({"default": nn.Linear(base.in_features, r, bias=False)})
            self.lora_B = nn.ModuleDict({"default": nn.Linear(r, base.out_features, bias=False)})
            nn.init.kaiming_uniform_(self.lora_A["default"].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B["default"].weight)
            self.scaling = alpha / r

        def forward(self, x):
            delta = self.lora_B["default"](self.lora_A["default"](self.lora_dropout["default"](x)))
            return self.base_layer(x) + delta * self.scaling

    class LoraModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

    class PeftModel(nn.Module):
        def __init__(self, model, cfg):
            super().__init__()
            for p in model.parameters():
                p.requires_grad_(False)
            wanted = set(cfg.target_modules)
            for name, mod in list(model.named_modules()):
                if name in wanted:
                    parent, leaf = name.rsplit(".", 1)
                    setattr(model.get_submodule(parent), leaf, LoraLinear(mod, cfg.r, cfg.lora_alpha, cfg.lora_dropout))
            self.base_model = LoraModel(model)
            self.peft_config = {"default": cfg}

        def save_pretrained(self, d, **kw):
            from safetensors.torch import save_file
            os.makedirs(d, exist_ok=True)
            save_file({n: p.detach().cpu().contiguous() for n, p in self.named_parameters() if p.requires_grad},
                      os.path.join(d, "adapter_model.safetensors"))
            cfg = self.peft_config["default"]
            with open(os.path.join(d, "adapter_config.json"), "w") as fh:
                json.dump({"r": cfg.r, "lora_alpha": cfg.lora_alpha, "lora_dropout": cfg.lora_dropout,
                           "target_modules": sorted(cfg.target_modules), "stub": True}, fh)

    mod = types.ModuleType("peft")
    mod.__version__ = "99.0.0+local-test-stub"
    mod.LoraConfig = LoraConfig
    mod.get_peft_model = lambda model, cfg: PeftModel(model, cfg)
    mod.prepare_model_for_kbit_training = lambda model, **kw: model
    return mod


def ensure_peft():
    """Use real peft if it is installed. Otherwise make a stand-in importable ONLY while train_lora's own
    peft-using functions run (attach_lora, package_versions), so transformers never sees a fake package."""
    if "peft_mode" in _CACHE:
        return
    import importlib.util
    if importlib.util.find_spec("peft") is not None:
        import peft
        _CACHE["peft_mode"] = FINDINGS["peft_used_in_tests"] = f"real peft {peft.__version__}"
        return
    stub = _build_peft_stub()

    def with_stub(fn):
        def wrapper(*a, **kw):
            sys.modules["peft"] = stub
            try:
                return fn(*a, **kw)
            finally:
                sys.modules.pop("peft", None)
        return wrapper

    tl.attach_lora = with_stub(tl.attach_lora)
    tl.package_versions = with_stub(tl.package_versions)
    _CACHE["peft_mode"] = FINDINGS["peft_used_in_tests"] = \
        "local stand-in (peft is not installed in the venv; it is on Kaggle that the real one runs)"


# ----------------------------------------------------------------------------------------------------------------
# tiny random Gemma 3 (same classes and module names as MedGemma; toy widths; NO pretrained weights)
# ----------------------------------------------------------------------------------------------------------------
def tiny_gemma3(text_layers, vision_layers):
    torch = torch_mod()
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration

    cfg = Gemma3Config.from_pretrained(MODEL_DIR)   # config.json only
    real_layers = cfg.text_config.num_hidden_layers
    t, v = cfg.text_config, cfg.vision_config
    t.hidden_size, t.intermediate_size, t.num_attention_heads, t.num_key_value_heads, t.head_dim = 32, 64, 2, 1, 16
    t.query_pre_attn_scalar = 16
    t.layer_types = list(t.layer_types)[:text_layers]
    t.num_hidden_layers = text_layers
    v.hidden_size, v.intermediate_size, v.num_attention_heads, v.num_hidden_layers = 16, 32, 2, vision_layers
    torch.manual_seed(0)
    model = Gemma3ForConditionalGeneration(cfg)
    return model.eval(), real_layers


@cached
def tiny_model_dir():
    """A loadable tiny model directory: random toy weights + links to the real processor files."""
    d = os.path.join(tmp_dir(), "tiny_model")
    model, _ = tiny_gemma3(text_layers=2, vision_layers=2)
    model.save_pretrained(d)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
              "tokenizer.model", "preprocessor_config.json", "processor_config.json", "chat_template.jinja"):
        src = os.path.join(MODEL_DIR, f)
        if os.path.exists(src):
            os.symlink(src, os.path.join(d, f))
    return d


@cached
def mini_bundle():
    """A bundle of 48 POOL tiles (8 per label x band stratum), in the layout train_lora.py expects."""
    t = project_tables()
    by = {}
    for r in t["pool_rows"]:
        by.setdefault(tl.stratum_of(r), []).append(r)
    chosen = []
    for k in sorted(by):
        names = tl.hash_order([r["image"] for r in by[k]], 1, "mini")[:8]
        chosen += [r for r in by[k] if r["image"] in set(names)]
    d = os.path.join(tmp_dir(), "bundle")
    os.makedirs(os.path.join(d, "gridded"))
    os.makedirs(os.path.join(d, "prompts"))
    shutil.copy(os.path.join(ROOT, "prompts", "rendered", "cte_p1.txt"), os.path.join(d, "prompts", "cte_p1.txt"))
    with open(os.path.join(d, "train_manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "label", "ssa_votes"])
        w.writeheader()
        for r in sorted(chosen, key=lambda r: r["image"]):
            assert r["image"] not in t["dev"] and r["image"] not in t["test"]
            w.writerow(r)
            with open(os.path.join(d, "gridded", r["image"]), "wb") as out:
                out.write(gridded_png_bytes(r["image"]))
    return d


def real_weights_guard():
    """Fail loudly if anything tries to load a model from the real MedGemma directory."""
    real = os.path.realpath(MODEL_DIR)
    original = tl.load_model

    def guarded(args, model_dir, prec):
        assert os.path.realpath(model_dir) != real, "the tests must never load the real weights"
        return original(args, model_dir, prec)

    tl.load_model = guarded


real_weights_guard()


# ----------------------------------------------------------------------------------------------------------------
class SplitAndSchedule(unittest.TestCase):
    def test_c_real_pool_split(self):
        t = project_tables()
        rows = t["pool_rows"]
        names = [r["image"] for r in rows]
        self.assertEqual(len(rows), tl.EXPECTED_POOL_N)
        self.assertEqual(tl.pool_sha256(names), tl.EXPECTED_POOL_SHA256)
        self.assertFalse(set(names) & t["dev"], "a dev tile is in the pool")
        self.assertFalse(set(names) & t["test"], "a test tile is in the pool")
        split = tl.make_split(rows)
        train, val = set(split["train"]), set(split["val"])
        self.assertFalse(train & val)                                     # disjoint
        self.assertEqual(train | val, set(names))                         # nothing lost, nothing added
        self.assertFalse((train | val) & (t["dev"] | t["test"]))          # never dev or test
        self.assertEqual(len(val), round(0.15 * len(rows)))               # 281
        self.assertEqual(split["seed"], 20261004)
        self.assertEqual(hashlib.sha256("\n".join(split["val"]).encode()).hexdigest(), tl.EXPECTED_VAL_SHA256)
        # deterministic: independent of row order and of repetition
        again = tl.make_split(list(reversed(rows)))
        self.assertEqual((again["train"], again["val"]), (split["train"], split["val"]))
        import random
        shuffled = rows[:]
        random.Random(5).shuffle(shuffled)
        self.assertEqual(tl.make_split(shuffled)["val"], split["val"])
        # stratified: every label x band stratum gives floor or ceil of 15%
        by = {}
        for r in rows:
            by.setdefault(tl.stratum_of(r), []).append(r["image"])
        self.assertEqual(len(by), 6)
        table = {}
        for k, members in by.items():
            n_val = len(val & set(members))
            q = 0.15 * len(members)
            self.assertIn(n_val, (math.floor(q), math.ceil(q)), k)
            self.assertEqual(split["strata"][k], {"n": len(members), "n_val": n_val, "n_train": len(members) - n_val})
            table[k] = f"{n_val}/{len(members)} = {n_val / len(members):.3f}"
        # a different seed gives a different slice (the seed matters)
        self.assertNotEqual(tl.make_split(rows, seed=1)["val"], split["val"])
        label = {r["image"]: r["label"] for r in rows}
        train_rows = [r for r in rows if r["image"] in train]
        weights, counts = tl.class_weights(train_rows)
        self.assertAlmostEqual(sum(weights[r["label"]] for r in train_rows) / len(train_rows), 1.0, places=12)
        self.assertAlmostEqual(weights["HP"] * counts["HP"], weights["SSA"] * counts["SSA"], places=9)
        FINDINGS["split"] = {"pool": len(rows), "train": len(train), "val": len(val),
                             "val_per_stratum": table, "train_counts": counts,
                             "val_counts": {c: sum(label[n] == c for n in val) for c in tl.CLASSES},
                             "class_weights": {c: round(w, 4) for c, w in weights.items()},
                             "val_names_sha256_16": hashlib.sha256("\n".join(split["val"]).encode()).hexdigest()[:16]}

    def test_c_synthetic_strata(self):
        rows = []
        for label, votes, n in (("HP", 0, 7), ("HP", 2, 13), ("HP", 3, 3), ("SSA", 4, 1), ("SSA", 6, 9), ("SSA", 7, 21)):
            rows += [{"image": f"T_{label}_{votes}_{i}.png", "label": label, "ssa_votes": votes} for i in range(n)]
        split = tl.make_split(rows)
        self.assertEqual(len(split["val"]), int(math.floor(0.15 * len(rows) + 0.5)))
        self.assertFalse(set(split["val"]) & set(split["train"]))
        for k, s in split["strata"].items():
            self.assertIn(s["n_val"], (math.floor(0.15 * s["n"]), math.ceil(0.15 * s["n"])), k)

    def test_manifest_validation(self):
        d = tmp_dir()

        def write(name, header, lines):
            p = os.path.join(d, name)
            with open(p, "w") as fh:
                fh.write(header + "\n" + "\n".join(lines) + "\n")
            return p

        good = write("m_good.csv", "image,label,ssa_votes", ["B.png,SSA,5", "A.png,HP,1"])
        self.assertEqual([r["image"] for r in tl.read_manifest(good)], ["A.png", "B.png"])
        for name, header, lines in (
                ("m_contradiction.csv", "image,label,ssa_votes", ["A.png,HP,6"]),
                ("m_duplicate.csv", "image,label,ssa_votes", ["A.png,HP,1", "A.png,HP,1"]),
                ("m_label.csv", "image,label,ssa_votes", ["A.png,TA,1"]),
                ("m_column.csv", "image,label", ["A.png,HP"]),
                ("m_test_tile.csv", "image,label,ssa_votes,partition", ["A.png,HP,1,test"])):
            with self.assertRaises(SystemExit, msg=name):
                tl.read_manifest(write(name, header, lines))

    def test_epoch_groups_and_schedule(self):
        torch = torch_mod()
        names = [f"T{i:04d}.png" for i in range(1594)]
        g0, g0b, g1 = tl.epoch_groups(names, tl.SEED, 0), tl.epoch_groups(names, tl.SEED, 0), tl.epoch_groups(names, tl.SEED, 1)
        self.assertEqual(g0, g0b)
        self.assertNotEqual(g0, g1)
        for g in (g0, g1):
            self.assertEqual(sorted(n for grp in g for n in grp), names)       # each epoch is a permutation
            self.assertEqual(len(g), 200)
            self.assertTrue(all(len(grp) == 8 for grp in g[:-1]) and len(g[-1]) == 2)
        total, warm = 800, tl.warmup_steps_for(800)
        self.assertEqual(warm, 40)
        from transformers import get_cosine_schedule_with_warmup
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=tl.LEARNING_RATE)
        sched = get_cosine_schedule_with_warmup(opt, warm, total)
        for step in range(total):
            self.assertAlmostEqual(opt.param_groups[0]["lr"], tl.lr_at(step, total, warm), places=15)
            opt.step()
            sched.step()
        self.assertEqual(tl.lr_at(0, total, warm), 0.0)
        self.assertAlmostEqual(tl.lr_at(warm, total, warm), 2e-4)
        FINDINGS["schedule"] = {"optimizer_steps": total, "warmup_steps": warm,
                                "identical_to": "transformers.get_cosine_schedule_with_warmup",
                                "lr_step_0": 0.0, "lr_peak_at_step": warm}

    def test_early_failure_keeps_a_nonzero_exit(self):
        """--exit-zero only applies when there is a checkpoint to preserve; a wrong path is a plain error."""
        out = os.path.join(tmp_dir(), "early_failure")
        with self.assertRaises(SystemExit) as cm:
            tl.run(["--model-dir", os.path.join(tmp_dir(), "no_such_model"), "--data-dir", os.path.join(tmp_dir(), "no_such_bundle"),
                    "--out-dir", out, "--device", "cpu", "--exit-zero", "yes", "--resume-from", "none"])
        self.assertNotIn(cm.exception.code, (0, None))
        with open(os.path.join(out, "FAILED.txt")) as fh:
            self.assertIn("not found", fh.read())

    def test_metrics(self):
        recs = [{"label": "HP", "pred": "HP", "score_HP": -0.1, "score_SSA": -3.0}] * 3 + \
               [{"label": "HP", "pred": "SSA", "score_HP": -2.0, "score_SSA": -0.2}] + \
               [{"label": "SSA", "pred": "SSA", "score_HP": -2.0, "score_SSA": -0.2}] + \
               [{"label": "SSA", "pred": "HP", "score_HP": -0.3, "score_SSA": -1.5}]
        m = tl.val_metrics(recs)
        self.assertAlmostEqual(m["recall_HP"], 0.75)
        self.assertAlmostEqual(m["recall_SSA"], 0.5)
        self.assertAlmostEqual(m["balanced_accuracy"], 0.625)
        self.assertAlmostEqual(m["accuracy"], 4 / 6, places=5)
        # expected accuracy under tier-1 sampling (report only): mean two-way probability of the true label, with
        # p >= 0.95 counted as 1 and p <= 0.05 as 0 (top_p 0.95 removes a label below 5%)
        sig = lambda a, b: 1 / (1 + math.exp(b - a))                           # noqa: E731
        p_hp = [sig(-0.1, -3.0)] * 3 + [sig(-2.0, -0.2)]                       # 0.948 (just under the 0.95 cut) and 0.142
        self.assertTrue(0.05 < p_hp[0] < 0.95 and 0.05 < p_hp[3] < 0.95)
        p_ssa = [sig(-0.2, -2.0), sig(-1.5, -0.3)]
        self.assertAlmostEqual(m["expected_tier1_accuracy"], (sum(p_hp) + sum(p_ssa)) / 6, places=5)
        self.assertAlmostEqual(m["expected_tier1_balanced_accuracy"], (sum(p_hp) / 4 + sum(p_ssa) / 2) / 2, places=5)
        sure = [{"label": "HP", "pred": "HP", "score_HP": -0.01, "score_SSA": -6.0},      # p = 0.9975 -> 1
                {"label": "SSA", "pred": "HP", "score_HP": -0.01, "score_SSA": -6.0},     # p = 0.0025 -> 0
                {"label": "SSA", "pred": "SSA", "score_HP": -0.7, "score_SSA": -0.7}]     # p = 0.5
        m2 = tl.val_metrics(sure)
        self.assertAlmostEqual(m2["expected_tier1_accuracy"], (1 + 0 + 0.5) / 3, places=6)
        self.assertAlmostEqual(m2["expected_tier1_balanced_accuracy"], (1.0 + 0.25) / 2, places=6)
        self.assertAlmostEqual(m2["balanced_accuracy"], 0.75)                  # argmax: HP 1/1, SSA 1/2

    def test_kernel_start_keeps_injected_arguments(self):
        """kaggle_push's header + train_lora.parse_args, started as a plain script and by a notebook kernel."""
        import contextlib
        import kaggle_push as kp
        with open(tl.__file__) as fh:
            source = fh.read()
        self.assertNotIn("sys.exit(run())", source)                # a kernel reports sys.exit(0) as an error
        self.assertIn("_exit_code = run()", source)
        injected = ["--verify-only", "--model-dir", "/kaggle/input/w", "--data-dir", "/kaggle/input/b", "--debug-limit-train", "32"]
        pushed, header = kp.staged_script(source, "train_lora.py", {}, injected)
        self.assertTrue(pushed.startswith(source[:200]) and header in pushed)
        saved_argv, saved_env = sys.argv, {k: os.environ.get(k) for k in ("KAGGLE_PUSH_STARTED", "TRAIN_LORA_ARGS")}
        try:
            for argv in (["/kaggle/src/script.py"], ["/opt/conda/lib/python3.12/site-packages/ipykernel_launcher.py", "-f", "/tmp/kernel-1.json"],
                         ["/usr/local/lib/python3.12/dist-packages/ipykernel_launcher.py"], []):
                os.environ.pop("KAGGLE_PUSH_STARTED", None)
                os.environ.pop("TRAIN_LORA_ARGS", None)
                sys.argv = list(argv)
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(header, "<kaggle_push header>", "exec"), {})          # noqa: S102 - the header this repo generates
                self.assertEqual(sys.argv, ["train_lora.py"] + injected, argv)
                a = tl.parse_args()
                self.assertEqual((a.verify_only, a.model_dir, a.data_dir, a.debug_limit_train),
                                 (True, "/kaggle/input/w", "/kaggle/input/b", 32), argv)
            # a process the script starts itself keeps its own command line
            sys.argv = ["train_lora.py", "--out-dir", "/x"]
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(header, "<kaggle_push header>", "exec"), {})              # noqa: S102
            self.assertEqual(sys.argv, ["train_lora.py", "--out-dir", "/x"])
            # no header (a hand-made notebook): the kernel's own -f argument is not ours; TRAIN_LORA_ARGS supplies them
            os.environ.pop("KAGGLE_PUSH_STARTED", None)
            sys.argv = ["ipykernel_launcher.py", "-f", "/tmp/kernel-2.json"]
            self.assertFalse(tl.parse_args().verify_only)
            os.environ["TRAIN_LORA_ARGS"] = "--verify-only --out-dir '/kaggle/working/a b'"
            a = tl.parse_args()
            self.assertEqual((a.verify_only, a.out_dir), (True, "/kaggle/working/a b"))
            sys.argv = ["ipykernel_launcher.py", "--verify-only"]                       # arguments that ARE there are used
            os.environ.pop("TRAIN_LORA_ARGS", None)
            self.assertTrue(tl.parse_args().verify_only)
        finally:
            sys.argv = saved_argv
            for k, v in saved_env.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        # the same header keeps infer_jobs.py's arguments
        import infer_jobs as ij
        with open(ij.__file__) as fh:
            _, header2 = kp.staged_script(fh.read(), "infer_jobs.py", {}, ["--jobs", "/kaggle/input/j/x.jsonl", "--batch-size", "2"])
        saved_argv, saved_flag = sys.argv, os.environ.pop("KAGGLE_PUSH_STARTED", None)
        try:
            sys.argv = ["ipykernel_launcher.py", "-f", "/tmp/kernel-3.json"]
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(header2, "<kaggle_push header>", "exec"), {})             # noqa: S102
            b = ij.parse_args()
            self.assertEqual((b.jobs, b.batch_size), ("/kaggle/input/j/x.jsonl", 2))
        finally:
            sys.argv = saved_argv
            os.environ.pop("KAGGLE_PUSH_STARTED", None) if saved_flag is None else os.environ.__setitem__("KAGGLE_PUSH_STARTED", saved_flag)
        FINDINGS["kernel_start"] = {"pushed_header_sets_argv0_to_the_source_name": True,
                                    "verify_only_survives_an_ipykernel_start": True, "no_sys_exit_0": True}

    def test_gradient_checkpointing_memory_rule(self):
        t4 = {"compute_dtype": "float32", "gpu": {"memory_gb": 14.56}}
        self.assertIn("12 GB", tl.gradient_checkpointing_problem(True, t4))
        self.assertIsNone(tl.gradient_checkpointing_problem(False, t4))
        self.assertIsNone(tl.gradient_checkpointing_problem(True, {"compute_dtype": "float32", "gpu": {"memory_gb": 24.0}}))
        self.assertIsNone(tl.gradient_checkpointing_problem(True, {"compute_dtype": "bfloat16", "gpu": {"memory_gb": 14.56}}))
        self.assertIsNone(tl.gradient_checkpointing_problem(True, {"compute_dtype": "float32", "gpu": None}))   # CPU tests


class OfficialFormat(unittest.TestCase):
    def test_a_prompt_identical_to_apply_chat_template(self):
        torch = torch_mod()
        b = builder()
        name, image = pool_tile()
        ids, pix = b.encode_prompt(image)
        ref = processor().apply_chat_template(tl.user_messages(prompt_text(), image), add_generation_prompt=True,
                                              tokenize=True, return_dict=True, return_tensors="pt")
        self.assertEqual(ids.tolist(), ref["input_ids"][0].tolist())
        self.assertEqual(len(ids), tl.EXPECTED_PROMPT_TOKENS)                 # 822, as in PLAN amendment 2
        self.assertTrue(torch.equal(pix, ref["pixel_values"][0]))
        self.assertEqual(tuple(pix.shape), (3, 896, 896))
        self.assertEqual(int((ids == b.image_token_id).sum()), 256)
        ex = b.training_example(image, "SSA")
        self.assertEqual(ex["input_ids"][: len(ids)].tolist(), ref["input_ids"][0].tolist())
        self.assertEqual(ex["token_type_ids"][: len(ids)].tolist(), ref["token_type_ids"][0].tolist())
        tok = b.tok
        self.assertEqual(tok.convert_ids_to_tokens(ids[:5].tolist()),
                         ["<bos>", "<start_of_turn>", "user", "\n\n\n", "<start_of_image>"])   # image first
        self.assertEqual(tok.convert_ids_to_tokens(ids[-3:].tolist()), ["<start_of_turn>", "model", "\n"])
        self.assertEqual(ids.tolist().count(tok.bos_token_id), 1)             # exactly one <bos>
        self.assertNotIn("system", tok.decode(ids[:8]))
        # what Google's notebook collate would do (special tokens added again) is NOT identical: a second <bos>
        twice = processor()(text=[b.template_text], images=[[image]], return_tensors="pt")["input_ids"][0]
        self.assertEqual(len(twice), len(ids) + 1)
        self.assertEqual(twice[:2].tolist(), [tok.bos_token_id, tok.bos_token_id])
        # a second pool tile gives the same prompt ids (only the pixels differ)
        from PIL import Image
        other_name = project_tables()["pool_rows"][1]["image"]
        other = Image.open(io.BytesIO(gridded_png_bytes(other_name))).convert("RGB")
        ids2, pix2 = b.encode_prompt(other)
        self.assertEqual(ids2.tolist(), ids.tolist())
        self.assertFalse(torch.equal(pix, pix2))
        FINDINGS["a_prompt"] = {"tile": name, "prompt_tokens": len(ids), "image_tokens": 256,
                                "token_identical_to_apply_chat_template": True, "pixel_values_identical": True,
                                "image_processor": type(processor().image_processor).__name__,
                                "notebook_style_collate_adds_second_bos": True}

    def test_b_labels_only_on_target(self):
        torch = torch_mod()
        b = builder()
        _, image = pool_tile()
        n_prompt = tl.EXPECTED_PROMPT_TOKENS
        for label in tl.CLASSES:
            ex = b.training_example(image, label)
            ids, labels = ex["input_ids"], ex["labels"]
            t = len(b.target_ids[label])
            self.assertEqual(len(ids), n_prompt + t)
            self.assertTrue(bool((labels[:n_prompt] == -100).all()))              # nothing on the prompt
            self.assertTrue(bool((labels[ids == b.image_token_id] == -100).all()))  # nothing on image tokens
            self.assertEqual(labels[n_prompt:].tolist(), b.target_ids[label])     # exactly the target
            self.assertEqual(ids[n_prompt:].tolist(), b.target_ids[label])
            self.assertEqual(int((labels != -100).sum()), t)
            self.assertEqual(b.tok.decode(labels[labels != -100]), tl.TARGET_TEMPLATE.format(label=label))
            self.assertEqual(int(ex["token_type_ids"].sum()), 256)
            self.assertTrue(bool(ex["attention_mask"].all()))
        n_t = N_TARGET[tl.TARGET_VARIANT]
        self.assertEqual(len(b.target_ids["HP"]), n_t)
        batch = tl.collate([b.training_ids("HP"), b.training_ids("SSA")], b.pad_id, torch)
        self.assertEqual(tuple(batch["input_ids"].shape), (2, n_prompt + n_t))    # same length: no padding at all
        self.assertTrue(bool(batch["attention_mask"].all()))
        self.assertEqual(tl.trailing_supervised(batch), n_t)
        FINDINGS["b_labels"] = {"target_variant": tl.TARGET_VARIANT, "sequence_length": n_prompt + n_t,
                                "supervised_tokens_per_example": n_t,
                                "labels_minus100_elsewhere": True, "padding_needed": False}

    def test_d_label_tokenisation(self):
        b = builder()
        tok = b.tok
        self.assertEqual(tl.TARGET_TEMPLATES["plan"], '{{\n  "label": "{label}"')    # the PLAN's assistant text, verbatim
        self.assertEqual(tl.TARGET_TEMPLATE, tl.TARGET_TEMPLATES[tl.TARGET_VARIANT])
        # both variants, whatever the file is set to: "plan" ends in the bare quote token, "label_end" at the label
        prefix = ["{", "\n", "▁▁", '"', "label", '":', '▁"']
        for variant, tail in (("plan", ['"']), ("label_end", [])):
            ids = {c: tok(tl.TARGET_TEMPLATES[variant].format(label=c), add_special_tokens=False)["input_ids"] for c in tl.CLASSES}
            for c in tl.CLASSES:
                self.assertEqual(tok.convert_ids_to_tokens(ids[c]), prefix + [c] + tail, variant)
                self.assertEqual(len(ids[c]), N_TARGET[variant])
            other = tl.label_scoring_plan(ids)
            self.assertEqual((other["mode"], len(other["common_prefix_ids"]), other["common_suffix_len"]),
                             ("single_token_logits", 7, len(tail)), variant)
            self.assertEqual(tok.convert_ids_to_tokens([other["scored_ids"]["HP"][0], other["scored_ids"]["SSA"][0]]), ["HP", "SSA"])
        bare_quote, quote_comma = tok.convert_tokens_to_ids('"'), tok.convert_tokens_to_ids('",')
        self.assertEqual((bare_quote, quote_comma), (236775, 827))
        toks = {c: tok.convert_ids_to_tokens(b.target_ids[c]) for c in tl.CLASSES}
        closing = ['"'] if tl.TARGET_VARIANT == "plan" else []
        self.assertEqual(toks["HP"], prefix + ["HP"] + closing)
        self.assertEqual(toks["SSA"], prefix + ["SSA"] + closing)
        plan = b.scoring
        self.assertEqual(plan["mode"], "single_token_logits")
        self.assertEqual(len(plan["common_prefix_ids"]), 7)
        self.assertEqual(plan["common_suffix_len"], len(closing))                 # "plan": the closing quote, its own token
        hp, ssa = plan["scored_ids"]["HP"], plan["scored_ids"]["SSA"]
        self.assertEqual((len(hp), len(ssa)), (1, 1))
        self.assertEqual(tok.convert_ids_to_tokens(hp + ssa), ["HP", "SSA"])
        self.assertNotEqual(hp, ssa)
        # tokenising prompt and target together gives the same ids as prompt ids + target ids (no merge at the seam)
        _, image = pool_tile()
        for c in tl.CLASSES:
            joint = processor()(text=[b.template_text + b.target_text[c]], images=[[image]], return_tensors="pt",
                                add_special_tokens=False)["input_ids"][0].tolist()
            self.assertEqual(joint, b.ref_prompt_ids.tolist() + b.target_ids[c])
        # the scoring input ends right before the label token
        s = b.scoring_ids()
        self.assertEqual(s["input_ids"].tolist(), b.ref_prompt_ids.tolist() + b.target_ids["HP"][:7])
        # in a full answer the quote after the label merges with the comma: '",' is one token, not '"' then ','
        full = tok.convert_ids_to_tokens(tok('{\n  "label": "HP",\n  "confidence": 0.9', add_special_tokens=False)["input_ids"])
        self.assertEqual(full[7:9], ["HP", '",'])
        FINDINGS["d_label_tokens"] = {
            "target_variant": tl.TARGET_VARIANT,
            "HP": {"ids": b.target_ids["HP"], "tokens": toks["HP"]}, "SSA": {"ids": b.target_ids["SSA"], "tokens": toks["SSA"]},
            "label_token_ids": {"HP": hp[0], "SSA": ssa[0]}, "labels_are_single_tokens": True,
            "shared_prefix_tokens": 7, "closing_quote_is_separate_token": True,
            "read_out": "two logits at the position after the 7-token prefix (one forward pass per tile)",
            "note": "in a full cte_p1 answer the model writes the token '\",' (id 827) after the label; the PLAN's "
                    "label-only target ('plan') ends in the bare '\"' token (id 236775) instead, which the full answer "
                    "never uses; 'label_end' stops at the label and needs a PLAN amendment before training"}

    def test_d_multi_token_fallback(self):
        plan = tl.label_scoring_plan({"HP": [1, 2, 3, 9], "SSA": [1, 2, 4, 5, 9]})
        self.assertEqual(plan["mode"], "sequence_log_prob")
        self.assertEqual(plan["common_prefix_ids"], [1, 2])
        self.assertEqual(plan["scored_ids"], {"HP": [3, 9], "SSA": [4, 5, 9]})   # label tokens through the terminator
        merged = tl.label_scoring_plan({"HP": [1, 7], "SSA": [1, 8]})            # quote merged into the label token
        self.assertEqual((merged["mode"], merged["scored_ids"]), ("single_token_logits", {"HP": [7], "SSA": [8]}))
        with self.assertRaises(SystemExit):
            tl.label_scoring_plan({"HP": [1, 2], "SSA": [1, 2, 3]})

    def test_loss_functions(self):
        torch = torch_mod()
        g = torch.Generator().manual_seed(0)
        b_, n, v, t = 3, 12, 50, 4
        logits = torch.randn(b_, n, v, generator=g)
        ids = torch.randint(0, v, (b_, n), generator=g)
        labels = torch.full((b_, n), -100)
        labels[:, -t:] = ids[:, -t:]
        general = tl.ce_general(torch, logits, labels)
        fast = tl.ce_trailing(torch, logits[:, -(t + 1):, :], ids[:, -t:])
        self.assertTrue(torch.allclose(general, fast, atol=1e-6))
        manual = torch.stack([torch.stack([-torch.log_softmax(logits[i, p - 1], -1)[ids[i, p]] for p in range(n - t, n)]).mean()
                              for i in range(b_)])
        self.assertTrue(torch.allclose(general, manual, atol=1e-6))
        batch = {"labels": labels, "attention_mask": torch.ones(b_, n, dtype=torch.long)}
        self.assertEqual(tl.trailing_supervised(batch), t)
        padded = {"labels": labels, "attention_mask": torch.cat([torch.ones(b_, n - 1), torch.zeros(b_, 1)], 1).long()}
        self.assertEqual(tl.trailing_supervised(padded), 0)                       # padded -> general path
        ragged = tl.collate([{"input_ids": torch.arange(5), "attention_mask": torch.ones(5, dtype=torch.long),
                              "token_type_ids": torch.zeros(5, dtype=torch.long), "labels": torch.tensor([-100, -100, -100, 3, 4])},
                             {"input_ids": torch.arange(7), "attention_mask": torch.ones(7, dtype=torch.long),
                              "token_type_ids": torch.zeros(7, dtype=torch.long), "labels": torch.tensor([-100] * 5 + [5, 6])}],
                            0, torch)
        self.assertEqual(ragged["input_ids"][0].tolist(), [0, 1, 2, 3, 4, 0, 0])   # right padding
        self.assertEqual(ragged["labels"][0].tolist(), [-100, -100, -100, 3, 4, -100, -100])
        self.assertEqual(tl.trailing_supervised(ragged), 0)


class ModuleTreeAndEngine(unittest.TestCase):
    def test_lora_targets_on_full_depth_tree(self):
        """Same classes, same depth (34 text / 27 vision blocks), toy widths: the module names are the real ones."""
        ensure_peft()
        torch = torch_mod()
        model, real_layers = tiny_gemma3(text_layers=34, vision_layers=27)
        self.assertEqual(real_layers, 34)
        targets, others = tl.select_lora_targets(model)
        tl.check_lora_targets(model, targets)
        self.assertEqual(len(targets), 34 * 7)
        for name in targets:
            parts = name.split(".")
            self.assertIn("language_model", parts)
            self.assertNotIn("vision_tower", parts)
            self.assertNotIn("multi_modal_projector", parts)
            self.assertIn(parts[-1], tl.LM_LINEAR_LEAVES)
        vision_linear = [n for n in others if "vision_tower" in n.split(".")]
        self.assertGreater(len(vision_linear), 0)                 # "all-linear" would have adapted these
        self.assertIn("lm_head", [n for n in others if "vision_tower" not in n.split(".")])
        every_lm_linear = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
                           and "language_model" in n.split(".")]
        self.assertEqual(sorted(every_lm_linear), sorted(targets))  # nothing in the language model is left out
        peft_model = tl.attach_lora(model, targets, torch)
        info = tl.verify_lora(peft_model, targets)
        self.assertEqual(info["n_target_modules"], 238)
        vt_name, vt = tl.find_submodule(model, "vision_tower")
        self.assertFalse(any("lora" in n.lower() for n, _ in vt.named_modules()))
        self.assertFalse(any(p.requires_grad for p in vt.parameters()))
        self.assertFalse(any(p.requires_grad for p in tl.find_submodule(model, "multi_modal_projector")[1].parameters()))
        self.assertTrue(all("lora_" in n for n, p in peft_model.named_parameters() if p.requires_grad))
        # 4-bit scope on the full-depth tree: the skip list isolates the language model under this transformers
        # version's own matching rule and under the 4.x substring rule; the check on a "loaded" model catches a
        # quantised vision-tower layer.
        fresh, _ = tiny_gemma3(text_layers=34, vision_layers=27)     # `model` is LoRA-wrapped by now: take an untouched tree
        linear = [n for n, m in fresh.named_modules() if type(m) is torch.nn.Linear]
        old_rule = [n for n in linear if not any((k + "." in n) or k == n for k in tl.BNB_SKIP_MODULES)]
        self.assertEqual(sorted(old_rule), sorted(targets))
        try:
            from transformers.quantizers.quantizers_utils import should_convert_module
            self.assertEqual(sorted(n for n in linear if should_convert_module(n, list(tl.BNB_SKIP_MODULES))), sorted(targets))
            alone = [n for n in linear if should_convert_module(n, ["vision_tower", "multi_modal_projector", "lm_head"])]
            skip_note = (f"skip list exact under the installed rule; 'vision_tower' alone would quantise "
                         f"{len(set(alone) - set(targets))} vision / head layers")
        except ImportError:
            skip_note = "installed transformers has no should_convert_module (substring rule)"
        import infer_jobs as ij
        self.assertEqual((tl.BNB_SKIP_MODULES, tl.QUANTIZED_SCOPE, tl.LM_LINEAR_LEAVES),
                         (ij.BNB_SKIP_MODULES, ij.QUANTIZED_SCOPE, ij.LM_LINEAR_LEAVES))    # trainer and evaluator agree

        class Linear4bit(torch.nn.Linear):                 # stands in for bitsandbytes' class (matched by name)
            pass

        mods = dict(fresh.named_modules())
        for n in targets:
            mods[n].__class__ = Linear4bit
        self.assertEqual(tl.check_quantized_scope(fresh), 238)
        vision_name = next(n for n in linear if "vision_tower" in n.split("."))
        mods[vision_name].__class__ = Linear4bit
        with self.assertRaises(SystemExit) as cm:
            tl.check_quantized_scope(fresh)
        self.assertIn("4-bit scope is wrong", str(cm.exception))
        mods[vision_name].__class__ = torch.nn.Linear
        mods["lm_head"].__class__ = Linear4bit
        with self.assertRaises(SystemExit):
            tl.check_quantized_scope(fresh)
        mods["lm_head"].__class__ = torch.nn.Linear
        mods[targets[0]].__class__ = torch.nn.Linear
        with self.assertRaises(SystemExit):                # 237 instead of 238
            tl.check_quantized_scope(fresh)
        FINDINGS["4bit_scope"] = {"quantised": "238 language-model linear layers", "vision_tower": "not quantised",
                                  "skip_list": skip_note, "checked_on_the_loaded_model": True}
        FINDINGS["lora_targets"] = {"n": len(targets), "per_block": list(tl.LM_LINEAR_LEAVES), "example": targets[0],
                                    "vision_tower_module": vt_name,
                                    "vision_linear_layers_left_alone": len(vision_linear),
                                    "other_linear_left_alone": [n for n in others if "vision_tower" not in n.split(".")]}

    def test_engine_on_tiny_model(self):
        ensure_peft()
        torch = torch_mod()
        b = builder()
        _, image = pool_tile()
        args = tl.parse_args(["--device", "cpu", "--precision", "float32"])
        prec = tl.resolve_precision(args, torch)
        model = tl.load_model(args, tiny_model_dir(), prec)
        targets, _ = tl.select_lora_targets(model)
        tl.check_lora_targets(model, targets)
        peft_model = tl.attach_lora(model, targets, torch)
        tl.verify_lora(peft_model, targets)
        with torch.no_grad():                                    # make the LoRA path matter (B is 0 at init)
            for n, p in peft_model.named_parameters():
                if "lora_B" in n:
                    p.normal_(std=0.05)
        engine = tl.Engine(torch, model, prec, b.image_token_id)
        self.assertTrue(engine.has_keep)
        pix = b.encode_image(image)
        engine.set_eval(peft_model)

        # 1. cached image features reproduce the pixel path
        score = {k: v.unsqueeze(0) for k, v in b.scoring_ids().items()}
        with torch.no_grad():
            feats = engine.image_features(pix.unsqueeze(0))
            via_pixels = engine.forward_logits(dict(score, pixel_values=pix.unsqueeze(0)), keep=1)
            via_cache = engine.forward_logits(dict(score, image_features=feats), keep=1)
        self.assertEqual(tuple(feats.shape), (1, 256, 32))
        self.assertTrue(torch.allclose(via_pixels, via_cache, atol=1e-5))

        # 2. the loss is the mean cross-entropy of exactly the 9 target tokens (independent recomputation
        #    from the hidden states, so an off-by-one in the logit window would show)
        ex = dict(b.training_ids("SSA"))
        batch = tl.collate([ex], b.pad_id, torch)
        with torch.no_grad():
            ce = engine.per_example_ce(dict(batch, image_features=feats))
            ce_pix = engine.per_example_ce(dict(batch, pixel_values=pix.unsqueeze(0)))
            hidden = model.model(input_ids=batch["input_ids"], pixel_values=pix.unsqueeze(0),
                                 attention_mask=batch["attention_mask"], token_type_ids=batch["token_type_ids"],
                                 use_cache=False).last_hidden_state
            n = batch["input_ids"].shape[1]
            n_t = N_TARGET[tl.TARGET_VARIANT]
            manual = torch.stack([-torch.log_softmax(model.lm_head(hidden[0, p - 1]).float(), -1)[batch["input_ids"][0, p]]
                                  for p in range(n - n_t, n)]).mean()
            # label read-out: logits at the position after the shared prefix
            last = model.lm_head(model.model(input_ids=score["input_ids"], pixel_values=pix.unsqueeze(0),
                                             attention_mask=score["attention_mask"],
                                             token_type_ids=score["token_type_ids"],
                                             use_cache=False).last_hidden_state[0, -1]).float()
        self.assertTrue(torch.allclose(ce, manual, atol=1e-4), (ce, manual))
        self.assertTrue(torch.allclose(ce_pix, manual, atol=1e-4))
        self.assertTrue(torch.allclose(via_pixels[0, -1], last, atol=1e-4))

        # 3. gradients reach LoRA matrices only, with and without gradient checkpointing (same values)
        def grads(checkpointing):
            if checkpointing:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                model.gradient_checkpointing_disable()
            engine.set_train(peft_model)
            self.assertFalse(engine.vision_tower.training)       # frozen vision tower stays in eval mode
            for p in peft_model.parameters():
                p.grad = None
            torch.manual_seed(3)
            engine.per_example_ce(dict(batch, image_features=feats)).sum().backward()
            return {n: p.grad.clone() for n, p in peft_model.named_parameters() if p.grad is not None}

        g_plain, g_ckpt = grads(False), grads(True)
        self.assertEqual(sorted(g_plain), sorted(g_ckpt))
        self.assertTrue(all("lora_A" in n or "lora_B" in n for n in g_plain))
        self.assertEqual(len(g_plain), 2 * len(targets))
        self.assertTrue(any(float(v.abs().sum()) > 0 for v in g_plain.values()))
        self.assertTrue(all(torch.allclose(g_plain[n], g_ckpt[n], atol=1e-6) for n in g_plain))
        FINDINGS["engine_tiny_model"] = {"cached_feature_path_equals_pixel_path": True,
                                         "loss_equals_manual_mean_ce_over_the_target_tokens": N_TARGET[tl.TARGET_VARIANT],
                                         "gradients_only_on_lora": True,
                                         "gradient_checkpointing_gives_same_gradients": True}


class EndToEnd(unittest.TestCase):
    """train_lora.main / run on the tiny model and a 48-tile pool bundle (CPU, float32, 2 debug epochs)."""

    def common(self):
        return ["--device", "cpu", "--precision", "float32", "--allow-pool-mismatch", "--allow-model-mismatch",
                "--debug-epochs", "2", "--feature-cache-dir", os.path.join(tmp_dir(), "feature_cache"),
                "--log-every-steps", "2", "--save-every-steps", "2", "--micro-batch", "4"]

    def argv(self, out, extra=()):
        return ["--model-dir", tiny_model_dir(), "--data-dir", mini_bundle(), "--out-dir", out,
                "--resume-from", "none", *self.common(), *extra]

    def run_main(self, out, extra=()):
        try:
            return tl.main(self.argv(out, extra))
        except SystemExit as e:
            return e.code

    def final_weights(self, out):
        return torch_mod().load(os.path.join(out, "checkpoints", "last", "trainable.pt"))

    def assert_same_weights(self, a, b):
        torch = torch_mod()
        self.assertEqual(sorted(a), sorted(b))
        self.assertTrue(all(torch.allclose(a[k], b[k], atol=1e-7) for k in a))
        return all(torch.equal(a[k], b[k]) for k in a)

    def test_full_run_resume_and_crash(self):
        ensure_peft()
        from safetensors.torch import load_file

        # frozen-pool guard: without the smoke-test flag a 48-tile manifest is refused
        with self.assertRaises(SystemExit) as cm:
            tl.load_bundle(mini_bundle())
        self.assertIn("frozen", str(cm.exception))
        self.assertEqual(self.run_main(os.path.join(tmp_dir(), "verify"), ["--verify-only"]), 0)

        # ---- A: uninterrupted ----------------------------------------------------------------------------------
        out_a = os.path.join(tmp_dir(), "run_a")
        self.assertEqual(self.run_main(out_a), 0)
        summary = read_json(os.path.join(out_a, "train_summary.json"))
        split = read_json(os.path.join(out_a, "split.json"))
        self.assertEqual(summary["status"], "complete")
        self.assertFalse(summary["protocol_run"])                      # debug flags -> never a protocol run
        self.assertEqual((len(split["train"]), len(split["val"])), (41, 7))
        self.assertFalse(set(split["train"]) & set(split["val"]))
        self.assertEqual(summary["epochs_completed"], 2)
        self.assertEqual(summary["hyperparameters"]["total_optimizer_steps"], 12)
        for key in ("chosen_epoch", "validation_per_epoch", "class_weights", "precision", "used_4bit",
                    "package_versions", "wall_time_seconds", "lora", "tokenization", "baseline_untrained_epoch0"):
            self.assertIn(key, summary)
        vals = summary["validation_per_epoch"]
        best = max(v["balanced_accuracy"] for v in vals)
        self.assertEqual(summary["chosen_epoch"], min(v["epoch"] for v in vals if v["balanced_accuracy"] == best))
        lines = read_jsonl(os.path.join(out_a, "train_log.jsonl"))
        self.assertEqual([(l["event"], l["epoch"]) for l in lines], [("baseline", 0), ("epoch", 1), ("epoch", 2)])
        for e in (1, 2):
            self.assertTrue(os.path.exists(os.path.join(out_a, "checkpoints", f"epoch_{e}", "adapter_model.safetensors")))
            scores = read_json(os.path.join(out_a, "val_scores", f"epoch_{e}.json"))
            self.assertEqual(sorted(t["image"] for t in scores["tiles"]), split["val"])   # validation tiles only
            redo = tl.val_metrics(scores["tiles"])
            self.assertEqual(redo["balanced_accuracy"], scores["metrics"]["balanced_accuracy"])
        best_file = os.path.join(out_a, "best_adapter", "adapter_model.safetensors")
        chosen_file = os.path.join(out_a, "checkpoints", f"epoch_{summary['chosen_epoch']}", "adapter_model.safetensors")
        self.assertEqual(tl.sha256_file(best_file), tl.sha256_file(chosen_file))
        self.assertTrue(os.path.exists(os.path.join(out_a, "best_adapter", "step4_meta.json")))
        # only the finished run's best_adapter is "final"; every epoch checkpoint stays non-final
        meta = read_json(os.path.join(out_a, "best_adapter", "step4_meta.json"))
        self.assertEqual((meta["final"], meta["chosen_epoch"], meta["epoch"], meta["epochs_completed"], meta["epochs_planned"]),
                         (True, summary["chosen_epoch"], summary["chosen_epoch"], 2, 2))
        self.assertEqual(meta["adapter_model_sha256"], tl.sha256_file(best_file))
        self.assertEqual(summary["best_adapter"], {"dir": "best_adapter", "adapter_model_sha256": meta["adapter_model_sha256"], "final": True})
        self.assertFalse(meta["protocol_run"])                         # ... and a debug run is never a protocol run
        for e in (1, 2):
            self.assertIs(read_json(os.path.join(out_a, "checkpoints", f"epoch_{e}", "step4_meta.json"))["final"], False)
        for v in vals:                                                  # report-only tier-1 expectation is in the log
            self.assertTrue(0.0 <= v["expected_tier1_accuracy"] <= 1.0 and 0.0 <= v["expected_tier1_balanced_accuracy"] <= 1.0)
        self.assertEqual(summary["eval_batch_final"], 4)
        self.assertEqual(summary["choices_not_fixed_by_the_addendum"]["max_grad_norm"], tl.MAX_GRAD_NORM)
        import infer_jobs as ij
        fa = ij.fingerprint_adapter(os.path.join(out_a, "best_adapter"))
        dev_job = [{"job_id": "dev__cte_p1__none__tier1__MHIST_xxx"}]
        with self.assertRaises(SystemExit):                             # final, but a debug run: not for dev tiles
            ij.adapter_gate(fa, dev_job)
        self.assertEqual(len(ij.adapter_gate(fa, [{"job_id": "smoke__cte_p1__none__tier1__MHIST_xxx"}])), 1)
        adapter = load_file(best_file)
        self.assertEqual(len(adapter), 2 * 2 * 7)                       # A and B for 7 linears in 2 tiny blocks
        self.assertTrue(all("language_model" in k and "vision_tower" not in k for k in adapter))
        self.assertTrue(any(float(v.abs().sum()) > 0 for k, v in adapter.items() if "lora_B" in k))  # it trained
        steps = read_jsonl(os.path.join(out_a, "train_steps.jsonl"))
        self.assertEqual(steps[0]["lr"], 0.0)
        self.assertTrue(all(math.isfinite(s["loss_weighted"]) for s in steps))
        final_a = self.final_weights(out_a)

        # already complete -> nothing is redone
        stamp = os.path.getmtime(os.path.join(out_a, "checkpoints", "last", "trainable.pt"))
        self.assertEqual(self.run_main(out_a), 0)
        self.assertEqual(os.path.getmtime(os.path.join(out_a, "checkpoints", "last", "trainable.pt")), stamp)
        # a changed configuration must not silently continue an old run
        self.assertNotEqual(self.run_main(out_a, ["--attn-implementation", "sdpa"]), 0)

        # ---- B: paused every 5 optimiser steps (mid-epoch) and resumed in place ---------------------------------
        out_b = os.path.join(tmp_dir(), "run_b")
        codes, paused_best = [], []
        for _ in range(6):
            codes.append(self.run_main(out_b, ["--debug-stop-after-steps", "5"]))
            if codes[-1] == 0:
                break
            paused = read_json(os.path.join(out_b, "train_summary.json"))
            self.assertEqual((paused["status"], paused["best_adapter"]["final"]), ("paused", False))
            meta_b = os.path.join(out_b, "best_adapter", "step4_meta.json")
            if os.path.exists(meta_b):                                  # best SO FAR of a paused run: never final
                paused_best.append(read_json(meta_b)["final"])
                with self.assertRaises(SystemExit):
                    tl.finalize_best_adapter(os.path.join(out_b, "best_adapter"), os.path.join(out_b, "checkpoints"),
                                             read_json(meta_b)["epoch"], paused["epochs_completed"], paused["epochs_planned"])
                self.assertIs(read_json(meta_b)["final"], False)
        self.assertEqual(codes, [tl.EXIT_PAUSED, tl.EXIT_PAUSED, 0])
        self.assertEqual(paused_best, [False])                          # second pause: epoch 1 validated, epoch 2 under way
        self.assertIs(read_json(os.path.join(out_b, "best_adapter", "step4_meta.json"))["final"], True)
        bitwise = self.assert_same_weights(final_a, self.final_weights(out_b))
        sum_b = read_json(os.path.join(out_b, "train_summary.json"))
        strip = lambda vs: [{k: v for k, v in d.items() if k != "seconds"} for d in vs]
        self.assertEqual(strip(sum_b["validation_per_epoch"]), strip(vals))
        self.assertEqual(sum_b["chosen_epoch"], summary["chosen_epoch"])
        self.assertEqual(sum_b["sessions"], 3)
        self.assertEqual(len(read_jsonl(os.path.join(out_b, "train_log.jsonl"))), 3)

        # ---- C: a crash in the middle of the epoch-1 validation, Kaggle exit convention, then a rerun ------------
        out_c = os.path.join(tmp_dir(), "run_c")
        original, calls = tl.val_metrics, {"n": 0}

        def flaky(records):
            calls["n"] += 1
            if calls["n"] == 2:                                     # 1 = baseline, 2 = epoch 1
                raise RuntimeError("simulated crash")
            return original(records)

        tl.val_metrics = flaky
        try:
            code = tl.run(self.argv(out_c, ["--exit-zero", "yes"]))
        finally:
            tl.val_metrics = original
        self.assertEqual(code, 0)                                   # output kept on Kaggle ...
        with open(os.path.join(out_c, "FAILED.txt")) as fh:         # ... and the failure is on record
            self.assertIn("simulated crash", fh.read())
        self.assertFalse(os.path.exists(os.path.join(out_c, "best_adapter")))
        calls["n"] = 0
        tl.val_metrics = flaky
        try:
            with self.assertRaises(RuntimeError):                   # without the Kaggle convention: a real error
                tl.run(self.argv(os.path.join(tmp_dir(), "run_c_strict"), ["--exit-zero", "no"]))
        finally:
            tl.val_metrics = original
        self.assertEqual(tl.run(self.argv(out_c, ["--exit-zero", "yes"])), 0)
        self.assertFalse(os.path.exists(os.path.join(out_c, "FAILED.txt")))
        self.assertTrue(os.path.exists(os.path.join(out_c, "FAILED.previous.txt")))
        self.assertEqual(read_json(os.path.join(out_c, "train_summary.json"))["status"], "complete")
        self.assert_same_weights(final_a, self.final_weights(out_c))
        log_c = read_jsonl(os.path.join(out_c, "train_log.jsonl"))
        self.assertEqual([(l["event"], l["epoch"]) for l in log_c], [("baseline", 0), ("epoch", 1), ("epoch", 2)])

        # ---- D: two Kaggle sessions: session 1 pauses (exit 0, status paused); session 2 has a fresh working
        #         directory, finds weights / bundle / earlier output under the input directory and continues ------
        out_d1 = os.path.join(tmp_dir(), "session1", "step4_lora")
        self.assertEqual(tl.run(self.argv(out_d1, ["--exit-zero", "yes", "--debug-stop-after-steps", "5"])), 0)
        self.assertEqual(read_json(os.path.join(out_d1, "train_summary.json"))["status"], "paused")
        fake_input = os.path.join(tmp_dir(), "kaggle_input")
        os.makedirs(os.path.join(fake_input, "earlier-kernel"))
        os.symlink(tiny_model_dir(), os.path.join(fake_input, "weights"))
        os.symlink(mini_bundle(), os.path.join(fake_input, "bundle"))
        os.symlink(out_d1, os.path.join(fake_input, "earlier-kernel", "step4_lora"))
        out_d2 = os.path.join(tmp_dir(), "session2", "step4_lora")
        saved_input, saved_env = tl.KAGGLE_INPUT, {k: os.environ.pop(k, None) for k in ("MEDGEMMA_DIR", "MHIST_FT_DATA_DIR", "MHIST_BUNDLE_DIR")}
        tl.KAGGLE_INPUT = fake_input
        try:
            code = tl.run(["--out-dir", out_d2, "--exit-zero", "yes", *self.common()])
        finally:
            tl.KAGGLE_INPUT = saved_input
            os.environ.update({k: v for k, v in saved_env.items() if v is not None})
        self.assertEqual(code, 0)
        sum_d = read_json(os.path.join(out_d2, "train_summary.json"))
        self.assertEqual((sum_d["status"], sum_d["sessions"], sum_d["chosen_epoch"]), ("complete", 2, summary["chosen_epoch"]))
        self.assert_same_weights(final_a, self.final_weights(out_d2))
        self.assertEqual(len(read_jsonl(os.path.join(out_d2, "train_log.jsonl"))), 3)
        self.assertEqual(tl.sha256_file(os.path.join(out_d2, "best_adapter", "adapter_model.safetensors")),
                         tl.sha256_file(best_file))

        # ---- E: a CUDA out-of-memory error in validation halves the eval batch and redoes it ------------------------
        torch = torch_mod()
        out_e = os.path.join(tmp_dir(), "run_e")
        original_forward, hits = tl.Engine.forward_logits, {"n": 0}

        def oom_once(self_, batch, keep=0):
            if "labels" not in batch and batch["input_ids"].shape[0] > 2:      # a validation batch of 4
                hits["n"] += 1
                raise torch.cuda.OutOfMemoryError("simulated")
            return original_forward(self_, batch, keep)

        tl.Engine.forward_logits = oom_once
        try:
            code = self.run_main(out_e)
        finally:
            tl.Engine.forward_logits = original_forward
        self.assertEqual(code, 0)
        sum_e = read_json(os.path.join(out_e, "train_summary.json"))
        self.assertEqual((sum_e["status"], sum_e["eval_batch_final"], hits["n"]), ("complete", 2, 1))
        self.assertTrue(any("out of memory in validation" in e["event"] for e in sum_e["events"]))
        self.assertEqual(sum_e["baseline_untrained_epoch0"]["n"], 7)             # every validation tile was still scored
        self.assert_same_weights(final_a, self.final_weights(out_e))             # training itself is untouched

        FINDINGS["end_to_end_tiny"] = {
            "split": "41 train / 7 validation of 48 pool tiles", "optimizer_steps": 12,
            "resumed_run_equals_uninterrupted_run": True, "bitwise_identical_after_resume": bitwise,
            "crash_during_validation_then_rerun_equals_uninterrupted": True,
            "kaggle_exit_convention": "pause / failure with a checkpoint -> exit 0, status in train_summary.json, FAILED.txt",
            "second_session_finds_inputs_and_earlier_output_and_matches": True,
            "finished_run_is_skipped": True, "changed_configuration_is_refused": True,
            "best_adapter_final_only_after_the_last_epoch": True, "validation_oom_halves_the_eval_batch": True,
            "validation_balanced_accuracy_per_epoch_random_tiny_model": [v["balanced_accuracy"] for v in vals]}


if __name__ == "__main__":
    unittest.main(verbosity=2 if "-v" in sys.argv else 1)
