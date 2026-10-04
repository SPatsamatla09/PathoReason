#!/usr/bin/env python3
r"""Build Kaggle job files for the dev evaluation of a configuration. LOCAL side; nothing is uploaded from here.

    python3 kaggle_ft/build_dev_jobs.py --config cte_p1 --tiles smoke            # 20 training-pool tiles, label-free checks
    python3 kaggle_ft/build_dev_jobs.py --config cte_p1 --control none noimage mismatch --tier 1 2 --step3-closed

STEP ORDER (PLAN, step-4 addendum): "the step-4 dev evaluation is run only if step 3 ends with no passing
configuration. If step 3 passes, step 3 is the winner whatever step 4 would have scored." Dev job files are
therefore built only with --step3-closed, which is your statement that step 3 has ended without a pass. Smoke
job files (pool tiles) need no such statement.

Both folders default to ~/mhist_local/ (outside the project, outside iCloud-synced Documents and outside
anything sync_to_pathoreason.py walks): job files in kaggle_jobs/, label sidecars in kaggle_local_only/.

For every config x control x tier it writes:

    <jobs-dir>/<name>.jsonl                UPLOAD (private). One job per tile: job_id, parts, tier, seed,
                                           max_new_tokens. No label, no vote count, no class assignment, no metadata.
    <local-dir>/<name>.sidecar.jsonl       LOCAL ONLY, never uploaded. job_id -> image, image_shown, label_true,
    <local-dir>/<name>.meta.json           ssa_votes, class_a_is, config, control, tier, tiles, prompt_sha256, plus
                                           the expected prompt token count and token-id hash; the job file's sha256.

    <name> = <tiles>__<config>__<control>__tier<tier>, e.g. dev__cte_p1__none__tier1

Images are not copied. A job names them as gridded/MHIST_xxx.png, relative to the data bundle that
build_bundle.py builds (--bundle-dir), together with the sha256 of the pipeline's own render (the exact PNG bytes
comp_run.py sends to the local model). If the bundle is on disk, every referenced file is compared with that
render and the build stops on a difference. infer_jobs.py checks the same sha256 on Kaggle.
--write-images writes missing tiles into <bundle-dir>/gridded/ (used by the tests to make a small bundle).

The content of each job is comp_run.py's own: the same prompt text (imported from comp_run), the same few-shot
examples and Class-A assignment (read from runs/competence/), the same seeded derangement for the mismatched
image control, and the same per-tile seed formula. kaggle_ft/test_jobs_local.py proves the equality against a
captured run of comp_run.main() and checks that the token ids match the official processor's.

It is deterministic and safe to re-run: an identical job file is left alone, a different one is refused unless
--force is given (a job file that was already uploaded must not change silently).

Test-partition tiles are never referenced: only the `dev` and `smoke` tile sets exist here, and every image a job
names must be in the train partition or the build stops. After writing, the jobs folder is checked with
build_bundle.audit_dir (no test tile, no dev tile paired with label information) when that module is present.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import comp_run as cr          # noqa: E402  prompt text, label tables, seeds: the single source of truth
import infer_jobs as ij        # noqa: E402  job-file validation and the token-id hash (stdlib-only at import)

LOCAL = os.environ.get("MHIST_LOCAL") or os.path.expanduser("~/mhist_local")
DEFAULT_JOBS_DIR = os.path.join(LOCAL, "kaggle_jobs")              # UPLOAD folder (private dataset, --kind jobs)
DEFAULT_LOCAL_DIR = os.path.join(LOCAL, "kaggle_local_only")       # label sidecars: never uploaded, never synced
PLAN_STEP_ORDER = ("PLAN, step-4 addendum: \"the step-4 dev evaluation is run only if step 3 ends with no passing "
                   "configuration. If step 3 passes, step 3 is the winner whatever step 4 would have scored.\"")


def in_synced_folder(path):
    """True if `path` is somewhere macOS may sync to iCloud, or inside the project tree that
    sync_to_pathoreason.py copies to the public repository (same rule as build_bundle.in_synced_folder)."""
    real = os.path.realpath(path)
    home = os.path.realpath(os.path.expanduser("~"))
    bad = [os.path.join(home, "Documents"), os.path.join(home, "Desktop"),
           os.path.join(home, "Library", "Mobile Documents"), os.path.join(home, "Library", "CloudStorage"),
           os.path.realpath(ROOT)]
    return any(real == b or real.startswith(b + os.sep) for b in bad)

CONFIGS = ["co_p1", "cte_p1", "neutral_cte", "names_crit_fs", "neutral_crit_fs"]
CONTROLS = ["none", "noimage", "mismatch"]
TIERS = ["1", "2"]
MAX_NEW_TOKENS = 1500
DEFAULT_PROCESSOR_DIR = os.environ.get("MEDGEMMA_DIR", os.path.expanduser("~/mhist_local/medgemma-1.5-4b-it"))
DEFAULT_BUNDLE_DIR = os.environ.get("MHIST_BUNDLE_DIR") or os.path.expanduser("~/mhist_local/kaggle_bundle_mhist")


def job_name(tiles, config, control, tier):
    return f"{tiles}__{config}__{control}__tier{tier}"


def job_seed(config, control, tier, img):
    """comp_run.py's per-tile seed, character for character."""
    return int(hashlib.sha256(f"{cr.SEED}:{config}:{control}:{tier}:{img}".encode()).hexdigest()[:8], 16) % (2 ** 31)


def derangement(base_set):
    """comp_run.py's mismatched-image map: a seeded shuffle of base_set, repeated until no tile keeps its own image."""
    perm = base_set[:]
    rng = random.Random(cr.SEED + 23)
    while True:
        rng.shuffle(perm)
        if all(a != b for a, b in zip(base_set, perm)):
            break
    return dict(zip(base_set, perm))


def prompt_for(config, a_is):
    if config in cr.PIPELINE_PROMPTS:
        return cr.PIPELINE_PROMPTS[config]
    if config in ("neutral_cte", "neutral_crit_fs"):
        return cr.prompt_neutral(a_is)
    return cr.prompt_names()


def content_parts(config, control, img, a_is, examples, mismatch):
    """The content list comp_run.main() builds for one call, with images named instead of embedded.

    Takes no label of the tile being classified. Returns (parts, prompt text, image shown or None)."""
    parts = []
    if examples:
        parts.append({"type": "text", "text": "Here are labelled example tiles of the two classes, drawn with the same grid."})
        for i, e in enumerate(examples, 1):
            if config.startswith("neutral"):
                name = "Class A" if e["label"] == a_is else "Class B"
            else:
                name = e["label"]
            parts.append({"type": "image", "image": e["image"]})
            parts.append({"type": "text", "text": f"Example {i}: {name}"})
        parts.append({"type": "text", "text": "Now the tile to classify:"})
    shown = None
    if control != "noimage":
        shown = mismatch.get(img, img)
        parts.append({"type": "image", "image": shown})
    prompt = prompt_for(config, a_is)
    parts.append({"type": "text", "text": prompt})
    return parts, prompt, shown


def hf_prompt_text(parts):
    """The string the official processor tokenizes for one user turn (PLAN amendment 2): text parts stripped and
    concatenated, each image as blank line + <start_of_image> + 256 soft tokens + <end_of_image> + blank line."""
    image = "\n\n<start_of_image>" + "<image_soft_token>" * ij.IMAGE_TOKENS_PER_IMAGE + "<end_of_image>\n\n"
    body = "".join(p["text"].strip() if p["type"] == "text" else image for p in parts)
    return "<bos><start_of_turn>user\n" + body + "<end_of_turn>\n<start_of_turn>model\n"


def load_tokenizer(processor_dir, mode):
    if mode == "off":
        return None
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(processor_dir, local_files_only=True)
        assert tok.convert_tokens_to_ids(ij.THINK_OPEN) == ij.THINK_OPEN_ID
        return tok
    except Exception as e:   # noqa: BLE001
        if mode == "require":
            sys.exit(f"STOP: --expect-tokens require, but the tokenizer in {processor_dir} cannot be loaded "
                     f"({type(e).__name__}: {e}). Run with ~/mhist_local/venv/bin/python.")
        print(f"note: no tokenizer here ({type(e).__name__}); the sidecar will not hold expected token counts and "
              "import_results.py will skip that check. Run with ~/mhist_local/venv/bin/python to add them.")
        return None


def load_context():
    """Everything frozen in runs/competence/. Stops if a frozen file is missing; this script never creates one."""
    for f in ("splits.json", "ab_assignment.json", "fewshot_examples.json"):
        if not os.path.exists(os.path.join(cr.OUT, f)):
            sys.exit(f"STOP: {os.path.join(cr.OUT, f)} is missing; it is frozen by comp_run.py and is not created here")
    lab, votes, part = cr.labels()
    splits = json.load(open(os.path.join(cr.OUT, "splits.json")))
    if set(splits["screen"]) | set(splits["dev_rest"]) != set(splits["dev"]) or set(splits["screen"]) & set(splits["dev_rest"]):
        sys.exit("STOP: splits.json: screen and dev_rest do not partition dev")
    return {"lab": lab, "votes": votes, "part": part, "splits": splits,
            "ab": cr.ab_assignment(), "examples": cr.fewshot_examples()}


def comp_run_order(tiles, ctx, config=""):
    """The order in which comp_run.py's local mode runs (and records) a tile list: as given, except that neutral
    few-shot configs are sorted by Class-A assignment so that the cached example prefix is reused."""
    if config.endswith("_fs") and config.startswith("neutral"):
        return sorted(tiles, key=lambda t: (ctx["ab"][t], t))
    return list(tiles)


def tile_list(which, ctx, config=""):
    """(tiles in run order, the list the mismatch derangement is drawn over, tiles-field of each tile).

    Run order is comp_run's local run order, so that the imported record files list the tiles as comp_run's own
    screen / dev_rest files do (comp_analyze's bootstrap resamples by position): the file order of splits.json,
    except that neutral few-shot configs are sorted by Class-A assignment, as comp_run's local mode does.
    For dev the derangement is always drawn over splits["dev"] in file order. For smoke comp_run draws it over
    its run list, so the same list is used here."""
    splits = ctx["splits"]

    def run_order(tiles):
        return comp_run_order(tiles, ctx, config)

    if which == "dev":
        screen = set(splits["screen"])
        tiles = run_order(splits["screen"]) + run_order(splits["dev_rest"])     # the screen subset runs first
        return tiles, splits["dev"], {t: ("screen" if t in screen else "dev_rest") for t in tiles}
    exs = {e["image"] for e in ctx["examples"]}                  # smoke: comp_run's 20 training-pool tiles
    tiles = run_order(random.Random(cr.SEED + 31).sample(sorted(set(splits["fewshot_pool"]) - exs), 20))
    return tiles, tiles, {t: "smoke" for t in tiles}


class ImageRefs:
    """Job-file reference for each gridded tile: its path inside the data bundle and the sha256 of its bytes.

    The bytes are the pipeline's own render (comp_run.b64_grid). If the bundle is on disk the file there must
    hold the same pixels; with write=True a missing file is written."""

    def __init__(self, bundle_dir, part, write=False):
        self.dir, self.part, self.write, self.refs = os.path.join(bundle_dir, "gridded"), part, write, {}
        self.checked = self.absent = self.written = 0

    def ref(self, image):
        if image in self.refs:
            return dict(self.refs[image])
        if self.part.get(image) != "train":
            sys.exit(f"STOP: {image} is in partition {self.part.get(image)!r}; only train-partition tiles may be referenced")
        png = base64.b64decode(cr.b64_grid(image))               # the exact bytes comp_run.py sends
        path = os.path.join(self.dir, image)
        if os.path.exists(path):
            have = open(path, "rb").read()
            if have != png:
                from PIL import Image
                a, b = Image.open(io.BytesIO(have)).convert("RGB"), Image.open(io.BytesIO(png)).convert("RGB")
                if a.size != b.size or a.tobytes() != b.tobytes():
                    sys.exit(f"STOP: {path} exists and its pixels differ from the pipeline's gridded render of {image}")
                png = have                                       # same pixels, other PNG encoding: keep the file
            self.checked += 1
        elif self.write:
            os.makedirs(self.dir, exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(png)
            self.written += 1
        else:
            self.absent += 1
        self.refs[image] = {"type": "image", "file": f"gridded/{image}", "sha256": hashlib.sha256(png).hexdigest()}
        return dict(self.refs[image])


def build(config, control, tier, which, ctx, image_refs, tokenizer=None, labels=None):
    """Returns (jobs, sidecar rows). `labels` = (label table, vote table) enters the sidecar rows ONLY."""
    lab, votes = labels if labels is not None else (ctx["lab"], ctx["votes"])
    tiles, base_set, tiles_field = tile_list(which, ctx, config)
    examples = ctx["examples"] if config.endswith("_fs") else []
    mismatch = derangement(base_set) if control == "mismatch" else {}
    name = job_name(which, config, control, tier)
    jobs, side, token_cache = [], [], {}
    for img in tiles:
        a_is = ctx["ab"][img]
        named_parts, prompt, shown = content_parts(config, control, img, a_is, examples, mismatch)
        parts = [p if p["type"] == "text" else image_refs.ref(p["image"]) for p in named_parts]
        job = ij.validate_job({"job_id": f"{name}__{os.path.splitext(img)[0]}", "parts": parts, "tier": tier,
                               "seed": job_seed(config, control, tier, img), "max_new_tokens": MAX_NEW_TOKENS})
        n_tok = ids_hash = None
        if tokenizer is not None:
            text = hf_prompt_text(parts)
            if text not in token_cache:
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                token_cache[text] = (len(ids), ij.ids_sha256(ids))
            n_tok, ids_hash = token_cache[text]
        jobs.append(job)
        side.append({"job_id": job["job_id"], "image": img, "image_shown": shown,
                     "label_true": lab[img], "ssa_votes": votes[img],
                     "class_a_is": a_is if config.startswith("neutral") else None,
                     "config": config, "control": control, "tier": tier, "tiles": tiles_field[img],
                     "seed": job["seed"], "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                     "fewshot": [e["image"] for e in examples] or None, "n_images": ij.n_images_of(job),
                     "max_new_tokens": MAX_NEW_TOKENS,
                     "n_prompt_tokens_expected": n_tok, "prompt_ids_sha256_expected": ids_hash})
    assert len({j["job_id"] for j in jobs}) == len(jobs)
    return jobs, side


def dumps_lines(rows):
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)


def write_if_changed(path, text, force, what):
    if os.path.exists(path):
        if open(path).read() == text:
            return "unchanged"
        if not force:
            sys.exit(f"STOP: {path} exists and differs from what would be built now. A {what} that may already be "
                     "uploaded is not overwritten silently; pass --force if you mean it.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)
    return "written"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", required=True, choices=CONFIGS)
    ap.add_argument("--control", nargs="+", default=["none"], choices=CONTROLS,
                    help="none | noimage (target tile omitted) | mismatch (another dev tile's image, seeded derangement)")
    ap.add_argument("--tier", nargs="+", default=["1"], choices=TIERS, help="1 = temperature 1.0, 2 = greedy")
    ap.add_argument("--tiles", default="dev", choices=["dev", "smoke"],
                    help="dev = the 300 dev tiles (screen first, then dev_rest), needs --step3-closed; smoke = comp_run's "
                         "20 training-pool tiles for label-free runtime checks (never scored). There is no test option.")
    ap.add_argument("--step3-closed", action="store_true",
                    help="required for --tiles dev: you state that step 3 has ENDED with no passing configuration. " + PLAN_STEP_ORDER)
    ap.add_argument("--jobs-dir", default=DEFAULT_JOBS_DIR,
                    help="UPLOAD folder for job files, nothing else goes in it (default: %(default)s)")
    ap.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR,
                    help="LOCAL-ONLY sidecars with labels; must be outside every upload folder, outside iCloud-synced "
                         "folders and outside the project tree (default: %(default)s)")
    ap.add_argument("--allow-synced-local-dir", action="store_true",
                    help="allow --local-dir inside Documents / Desktop / iCloud Drive / the project tree (NOT recommended: "
                         "the sidecars hold dev tile names with labels, and the project tree is copied to a public repository)")
    ap.add_argument("--bundle-dir", default=DEFAULT_BUNDLE_DIR,
                    help="data bundle built by build_bundle.py; its gridded/ tiles are compared with the pipeline's "
                         "render (default: %(default)s; env MHIST_BUNDLE_DIR)")
    ap.add_argument("--require-bundle", action="store_true",
                    help="stop if a referenced tile is not in <bundle-dir>/gridded/ (default: only report it)")
    ap.add_argument("--write-images", action="store_true",
                    help="write referenced tiles that are missing into <bundle-dir>/gridded/ (small test bundles)")
    ap.add_argument("--processor-dir", default=DEFAULT_PROCESSOR_DIR,
                    help="local copy of the model repo; only its tokenizer is read (default: %(default)s)")
    ap.add_argument("--expect-tokens", default="auto", choices=["auto", "require", "off"],
                    help="store the expected prompt token count and token-id hash in the sidecar (needs transformers)")
    ap.add_argument("--force", action="store_true", help="overwrite job files that differ")
    args = ap.parse_args(argv)

    if args.tiles == "dev" and not args.step3_closed:
        sys.exit("STOP: dev job files are built only after step 3 has ended with no passing configuration.\n  " + PLAN_STEP_ORDER
                 + "\n  If that is the case now, add --step3-closed. Until then only --tiles smoke (pool tiles) is available. "
                 "Nothing was written.")
    jobs_dir, local, bundle = (os.path.abspath(os.path.expanduser(d)) for d in (args.jobs_dir, args.local_dir, args.bundle_dir))
    if in_synced_folder(local) and not args.allow_synced_local_dir:
        sys.exit(f"STOP: --local-dir {local} is inside a folder macOS may sync to iCloud or inside the project tree, which "
                 "sync_to_pathoreason.py copies to a public repository. The sidecars pair dev tiles with their labels. "
                 f"Use the default ({DEFAULT_LOCAL_DIR}) or pass --allow-synced-local-dir.")
    for upload in (jobs_dir, bundle):
        if os.path.commonpath([upload, local]) in (upload, local):
            sys.exit(f"STOP: --local-dir ({local}) and the upload folder {upload} must not contain each other; "
                     "the sidecar holds labels and must never be uploaded")
    ctx = load_context()
    tokenizer = load_tokenizer(args.processor_dir, args.expect_tokens)
    image_refs = ImageRefs(bundle, ctx["part"], write=args.write_images)
    os.makedirs(local, exist_ok=True)
    marker = os.path.join(local, "DO_NOT_UPLOAD.txt")
    if not os.path.exists(marker):
        with open(marker, "w") as fh:
            fh.write("This folder holds dev labels (sidecar files). It stays on this machine. Never upload it.\n")
    for config in args.config:
        for control in args.control:
            for tier in args.tier:
                name = job_name(args.tiles, config, control, tier)
                jobs, side = build(config, control, tier, args.tiles, ctx, image_refs, tokenizer)
                jobs_text, side_text = dumps_lines(jobs), dumps_lines(side)
                for banned in ("label_true", "ssa_votes", "class_a_is", "image_shown", "Majority", "Annotators"):
                    if banned in jobs_text:
                        sys.exit(f"STOP: {name}: the job text contains {banned!r}; nothing written for it")
                jobs_path = os.path.join(jobs_dir, name + ".jsonl")
                state = write_if_changed(jobs_path, jobs_text, args.force, "job file")
                side_path = os.path.join(local, name + ".sidecar.jsonl")
                if tokenizer is None and os.path.exists(side_path):
                    # never downgrade: keep a sidecar that already holds the token expectations if nothing else differs
                    old = [json.loads(l) for l in open(side_path) if l.strip()]
                    blank = lambda rows: [{**r, "n_prompt_tokens_expected": None, "prompt_ids_sha256_expected": None} for r in rows]  # noqa: E731
                    if blank(old) == side and any(r["n_prompt_tokens_expected"] is not None for r in old):
                        print(f"{name}: {len(jobs)} jobs ({state}); sidecar kept (it already holds the token expectations)")
                        continue
                write_if_changed(side_path, side_text, True, "sidecar")
                meta = {"name": name, "config": config, "control": control, "tier": tier, "tiles": args.tiles,
                        "n_jobs": len(jobs), "jobs_file": os.path.basename(jobs_path),
                        "jobs_sha256": hashlib.sha256(jobs_text.encode()).hexdigest(),
                        "sidecar_sha256": hashlib.sha256(side_text.encode()).hexdigest(),
                        "comp_run_seed": cr.SEED, "max_new_tokens": MAX_NEW_TOKENS,
                        "images_referenced": len({p["file"] for j in jobs for p in j["parts"] if p["type"] == "image"}),
                        "expected_tokens": tokenizer is not None,
                        "prompt_tokens": sorted({s["n_prompt_tokens_expected"] for s in side}) if tokenizer else None}
                write_if_changed(os.path.join(local, name + ".meta.json"), json.dumps(meta, indent=1, sort_keys=True) + "\n",
                                 True, "meta file")
                print(f"{name}: {len(jobs)} jobs ({state}) -> {jobs_path}; "
                      f"{meta['images_referenced']} images; prompt tokens {meta['prompt_tokens']}; "
                      f"sidecar -> {os.path.join(local, name + '.sidecar.jsonl')} (local only)")
    r = image_refs
    print(f"images: {len(r.refs)} distinct tiles referenced; {r.checked} found in {r.dir} and identical to the "
          f"pipeline's render, {r.written} written, {r.absent} not on disk")
    if r.absent:
        msg = (f"{r.absent} referenced tiles are not in {r.dir}. Build the data bundle (kaggle_ft/build_bundle.py) "
               "before uploading; infer_jobs.py checks each tile's sha256 on Kaggle.")
        if args.require_bundle:
            sys.exit("STOP: " + msg)
        print("note: " + msg)
    try:
        import build_bundle
        problems = build_bundle.audit_dir(jobs_dir, allow=["*.jsonl"])     # job files and nothing else
    except ImportError:
        problems = None
        print("note: build_bundle.py not found; the jobs folder was not audited with its leak rules")
    if problems:
        sys.exit("STOP: the jobs folder fails build_bundle.audit_dir and must not be uploaded:\n  " + "\n  ".join(problems))
    print(f"jobs folder: {jobs_dir}" + (" (leak audit clean)" if problems is not None else "") +
          f". Upload ONLY this folder (PRIVATE dataset: kaggle_push.py dataset-create --kind jobs --dir {jobs_dir}). "
          f"Never upload {local}.")


if __name__ == "__main__":
    main()
