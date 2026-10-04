#!/usr/bin/env python3
"""Build the PRIVATE Kaggle data bundle for step 4 (runs/competence/PLAN.md, "Addendum: step 4").

Local and offline. Nothing is uploaded here; uploading is a separate, explicit step (kaggle_push.py).

    python3 kaggle_ft/build_bundle.py              build / refresh the bundle (default: ~/mhist_local/kaggle_bundle_mhist)
    python3 kaggle_ft/build_bundle.py --check      verify an existing bundle against its MANIFEST and the leak rules; writes nothing
    python3 kaggle_ft/build_bundle.py --out DIR    build somewhere else (must be OUTSIDE iCloud-synced folders)
    python3 kaggle_ft/build_bundle.py --prune      also delete files in the bundle directory that do not belong there

What goes in, and nothing else:

    gridded/<image>.png   the pipeline's gridded tile, rendered by run_experiment.b64_gridded_tile (byte-identical
                          to what comp_run.py sends), for the 1,875 fewshot_pool tiles and the 300 dev tiles
    train_manifest.csv    image,label,ssa_votes   -- fewshot_pool tiles ONLY
    dev_tiles.csv         image,subset            -- dev tile names and screen/dev_rest membership, NO labels
    prompts/*.txt         the pipeline prompts a job may reference (cte_p1.txt is the step-4 prompt)
    NOTICE.txt            private-use notice
    MANIFEST.json         counts, provenance and sha256/bytes of every file above

Asserted on every run (exit code 2 and a list of the problems if any fails):

    * no test-partition tile is rendered, named or referenced anywhere in the bundle
    * no dev tile carries a label or a vote count anywhere in the bundle (labels exist for pool tiles only)
    * the bundle directory holds exactly the files in MANIFEST.json (plus, at its top level only, kaggle_push.py's
      dataset-metadata.json and Finder's .DS_Store, neither of which the Kaggle CLI uploads from there)

Deterministic (sorted order, no timestamps, fixed renderer) and idempotent / resumable (a file whose bytes are
already right is not rewritten, so an interrupted build is finished by running the command again).

Importing this module needs the standard library only; Pillow / run_experiment are loaded when a build starts.
kaggle_push.py imports audit_dir() and verify_bundle() from here and runs them before every upload.
"""

import argparse
import base64
import csv
import fnmatch
import hashlib
import io
import json
import os
import re
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # grid_experiment/
ANNOTATIONS = os.path.join(ROOT, "..", "annotations.csv")
SPLITS = os.path.join(ROOT, "runs", "competence", "splits.json")
PROMPT_DIRS = (os.path.join(ROOT, "prompts", "rendered"), os.path.join(ROOT, "prompts", "rendered_competence"))
DEFAULT_OUT = os.environ.get("MHIST_BUNDLE_DIR") or os.path.join(os.path.expanduser("~"), "mhist_local", "kaggle_bundle_mhist")

EXPECT = {"fewshot_pool": 1875, "dev": 300, "screen": 100, "dev_rest": 200}   # PLAN.md, "Data, fixed once"
STEP4_PROMPT = "cte_p1.txt"
SCHEMA = 1
# Files that may sit at the TOP LEVEL of an upload directory without being part of it: the Kaggle metadata file
# that kaggle_push.py writes (the Kaggle CLI skips it there) and Finder litter (--ignore-patterns .DS_Store). The
# same names deeper down are NOT exempt: the CLI would put them into the zip of that sub-directory.
NOT_BUNDLE = {"dataset-metadata.json", ".DS_Store"}
# Never in any upload directory, whatever else it holds (build_dev_jobs.py's label sidecars and their marker).
FORBIDDEN_NAMES = ("DO_NOT_UPLOAD.txt", "*.sidecar.jsonl", "*.meta.json")
FORBIDDEN_DIRS = ("local_only", "kaggle_local_only")
ARCHIVE_EXT = (".zip", ".whl")
DEV_SUBSETS = ("screen", "dev_rest")

NOTICE = """PRIVATE WORKING COPY - DO NOT SHARE, DO NOT MAKE PUBLIC.

This directory holds image tiles derived from MHIST (Wei et al., 2021; Dartmouth-Hitchcock Medical Center),
used under the MHIST dataset research use agreement, with a 4x4 reference grid drawn on each tile.
It exists only so that its owner can run a private fine-tuning job on private cloud compute.

  - Keep every copy private: a private Kaggle dataset on the owner's account only, never public,
    never shared by link, never added to a public notebook's inputs.
  - It contains no test-partition tile and no label for any development tile.
  - Not for clinical use. No claim of diagnostic validity is made.
"""

TILE = re.compile(r"MHIST_[a-z]{3}")                           # a tile stem, with or without its .png
TEXT_EXT = {".csv", ".tsv", ".json", ".jsonl", ".txt", ".md", ".py", ".yaml", ".yml", ".jinja", ".cfg", ".ini",
            ".log", ".toml", ".sh", ""}
_TOKEN = r"(?<![A-Za-z0-9_]){}(?![A-Za-z0-9_])"
HP_TOK, SSA_TOK = re.compile(_TOKEN.format("HP")), re.compile(_TOKEN.format("SSA"))
LABEL_KEY = re.compile(r"(?i)label_true|true_label|gold_label|ssa_votes|n_ssa|majority|annotator|(?<![a-z])votes?(?![a-z])")
LABEL_VALUE = re.compile(r'"label"\s*:\s*"(?:HP|SSA)"(?!\s*or)')
LABEL_COLUMN = re.compile(r"(?i)label|vote|annotat|majority|ssa")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_truth():
    """Labels, votes, partition and the frozen splits. Read locally; only pool labels ever enter the bundle."""
    with open(ANNOTATIONS, newline="") as fh:
        rows = list(csv.DictReader(fh))
    lab = {r["Image Name"]: r["Majority Vote Label"] for r in rows}
    votes = {r["Image Name"]: int(r["Number of Annotators who Selected SSA (Out of 7)"]) for r in rows}
    part = {r["Image Name"]: r["Partition"] for r in rows}
    with open(SPLITS) as fh:
        splits = json.load(fh)
    return {"label": lab, "votes": votes, "partition": part, "splits": splits,
            "pool": set(splits["fewshot_pool"]), "dev": set(splits["dev"]),
            "test": {n for n, p in part.items() if p == "test"}}


def split_problems(t):
    """The frozen split must be what PLAN.md says before anything is rendered."""
    s, part, out = t["splits"], t["partition"], []
    for k, n in EXPECT.items():
        if len(s[k]) != n or len(set(s[k])) != n:
            out.append(f"splits.json: {k} has {len(s[k])} entries ({len(set(s[k]))} distinct), expected {n}")
    if t["pool"] & t["dev"]:
        out.append(f"splits.json: {len(t['pool'] & t['dev'])} tiles are in both fewshot_pool and dev")
    if set(s["screen"]) | set(s["dev_rest"]) != t["dev"] or set(s["screen"]) & set(s["dev_rest"]):
        out.append("splits.json: screen and dev_rest do not partition dev")
    for k, names in (("fewshot_pool", t["pool"]), ("dev", t["dev"])):
        bad = sorted(n for n in names if part.get(n) != "train")
        if bad:
            out.append(f"splits.json: {len(bad)} {k} tiles are not in the train partition, e.g. {bad[:3]}")
    if (t["pool"] | t["dev"]) & t["test"]:
        out.append("splits.json: a test-partition tile is in fewshot_pool or dev")
    for n in sorted(t["pool"]):
        if t["label"].get(n) not in ("HP", "SSA") or not 0 <= t["votes"].get(n, -1) <= 7:
            out.append(f"annotations.csv: bad label or vote count for pool tile {n}")
            break
    return out


# ---------------------------------------------------------------------------------------------------------------
# Leak tripwire: usable on ANY directory that is about to be uploaded (the bundle, a jobs folder, an adapter...).
# The structural guarantee is how the bundle is built; this is the independent second check.
# ---------------------------------------------------------------------------------------------------------------

def _is_text(path, rel, max_bytes):
    if os.path.splitext(rel)[1].lower() not in TEXT_EXT or os.path.getsize(path) > max_bytes:
        return False
    with open(path, "rb") as fh:
        return b"\0" not in fh.read(4096)


def audit_text(rel, text, truth, pool_labels_ok=True):
    """Tripwire for one text (a file's content, a script, a job file). Returns a list of violations.

    Line rule (every text): a line that names a dev tile must not carry label information.
    Table rule (.csv / .tsv): a table that names a dev tile must have exactly the header image,subset and, on
    every such row, exactly the tile name and screen / dev_rest. Any other table could code the label in a way
    no line rule can see (0/1, a column of its own).
    Document rule (.json): one JSON document is one unit, however it is indented. A document that names a dev
    tile must not contain a standalone HP / SSA token or a label-like key anywhere.
    """
    part, dev, pool = truth["partition"], truth["dev"], truth["pool"]
    low = rel.lower()
    lines = text.splitlines()
    header = lines[0] if lines else ""
    is_table = low.endswith((".csv", ".tsv"))
    sep = "\t" if low.endswith(".tsv") else ","
    label_csv = is_table and any(LABEL_COLUMN.search(c) for c in re.split(r"[,\t]", header))
    plain_dev_table = is_table and [c.strip() for c in header.split(sep)] == ["image", "subset"]

    def guarded(names):
        return any(n in dev or (not pool_labels_ok and n in pool) for n in names)

    n_test, n_label, n_table, first = 0, 0, 0, {}
    for i, line in enumerate(lines, 1):
        names = {m + ".png" for m in TILE.findall(line)}
        if not names:
            continue
        if any(part.get(n) == "test" for n in names):
            n_test += 1
            first.setdefault("test", i)
        if guarded(names):
            rest = TILE.sub("", line)
            lone = bool(HP_TOK.search(rest)) != bool(SSA_TOK.search(rest))
            if label_csv or lone or LABEL_KEY.search(rest) or LABEL_VALUE.search(rest):
                n_label += 1
                first.setdefault("dev", i)
            if is_table:
                cells = [c.strip() for c in line.split(sep)]
                if not (plain_dev_table and len(cells) == 2 and cells[0] in names and cells[1] in DEV_SUBSETS):
                    n_table += 1
                    first.setdefault("table", i)
    out = []
    if n_test:
        out.append(f"{rel}: {n_test} lines mention a test-partition tile (first at line {first['test']})")
    if n_label:
        out.append(f"{rel}: {n_label} lines pair a dev tile with label information (first at line {first['dev']})")
    if n_table:
        out.append(f"{rel}: {n_table} table rows name a dev tile, but the table is not the plain image,subset list "
                   f"(first at line {first['table']}); a table of dev tiles may hold nothing but the name and screen / dev_rest")
    if low.endswith(".json") and not n_label:
        named = {m + ".png" for m in TILE.findall(text)}
        if guarded(named):
            rest = TILE.sub("", text)
            what = [w for w, hit in (("a standalone HP / SSA token", HP_TOK.search(rest) or SSA_TOK.search(rest)),
                                     ("a label-like key", LABEL_KEY.search(rest))) if hit]
            if what:
                out.append(f"{rel}: the document names dev tiles and also contains {' and '.join(what)}; in a JSON "
                           "document the two need not be on one line to belong together")
    return out


def forbidden_path(rel):
    """Why this relative path may never be in an upload directory (None = no objection)."""
    parts = rel.replace(os.sep, "/").split("/")
    for d in parts[:-1]:
        if d in FORBIDDEN_DIRS:
            return f"it lies in a folder called {d} (local-only label sidecars)"
    for pat in FORBIDDEN_NAMES:
        if fnmatch.fnmatch(parts[-1], pat):
            return f"its name matches {pat} (build_dev_jobs.py's local-only files hold dev labels)"
    return None


def audit_archive(rel, full, truth, max_text_bytes, pool_labels_ok=True):
    """A zip / wheel: member names are checked, and every text member is audited like a file of its own."""
    out = []
    try:
        with zipfile.ZipFile(full) as z:
            names = z.namelist()
            hit = sorted({m + ".png" for m in TILE.findall(" ".join(names))} & truth["test"])
            if hit:
                out.append(f"{rel}: archive holds {len(hit)} test-partition tile names, e.g. {hit[0]}")
            for member in names:
                why = forbidden_path(member)
                if why:
                    out.append(f"{rel}!{member}: must never be uploaded: {why}")
                info = z.getinfo(member)
                if info.is_dir() or os.path.splitext(member)[1].lower() not in TEXT_EXT or info.file_size > max_text_bytes:
                    continue
                data = z.read(member)
                if b"\0" not in data[:4096]:
                    out += audit_text(f"{rel}!{member}", data.decode("utf-8", errors="replace"), truth, pool_labels_ok)
    except (zipfile.BadZipFile, OSError, RuntimeError, NotImplementedError):
        out.append(f"{rel}: unreadable zip archive")
    return out


def audit_dir(path, truth=None, max_text_bytes=64 << 20, pool_labels_ok=True, allow=None, only=None):
    """Return a list of violations (empty list = clean) for a directory destined for Kaggle.

    Rules:
      1. no file or archive member is named after a test-partition tile;
      2. no text file (or text member of a zip / wheel) mentions a test-partition tile at all;
      3. no text line that mentions a dev tile also carries label information (a lone HP / SSA token, a
         "label": "HP" style field, or a label / vote / annotator key); no table names a dev tile unless it is the
         plain image,subset list; no JSON document names a dev tile and holds label information anywhere;
      4. (pool_labels_ok=False only) the same as 3 for pool tiles;
      5. no local-only file of build_dev_jobs.py (DO_NOT_UPLOAD.txt, *.sidecar.jsonl, *.meta.json, a local_only
         folder) is present;
      6. (allow given: a list of fnmatch patterns over relative paths) every file must match one of them. That is
         how kaggle_push.py limits an upload to the files its kind may hold; a file whose content cannot be read
         as text (an archive, a pickle, a notebook, an unknown extension) is then only accepted by name.
    `only` (a set of relative paths) restricts the audit to those files. Symbolic links are followed; binary
    files are checked by name only. dataset-metadata.json and .DS_Store are exempt at the top level only.
    """
    t = truth or load_truth()
    out = []

    def test_tiles_in(text):
        return sorted({m + ".png" for m in TILE.findall(text)} & t["test"])

    for base, dirs, files in os.walk(path, followlinks=True):
        dirs.sort()
        for f in sorted(files):
            full = os.path.join(base, f)
            rel = os.path.relpath(full, path).replace(os.sep, "/")
            if f in NOT_BUNDLE and os.path.abspath(base) == os.path.abspath(path):
                continue
            if only is not None and rel not in only:
                continue
            hit = test_tiles_in(rel)
            if hit:
                out.append(f"{rel}: file is named after test-partition tile {hit[0]}")
            why = forbidden_path(rel)
            if why:
                out.append(f"{rel}: must never be uploaded: {why}")
            if allow is not None and not any(fnmatch.fnmatch(rel, pat) for pat in allow):
                out.append(f"{rel}: not a file this kind of upload may hold (allowed: {', '.join(allow)})")
                continue
            if rel.lower().endswith(ARCHIVE_EXT):
                out += audit_archive(rel, full, t, max_text_bytes, pool_labels_ok)
                continue
            if not _is_text(full, rel, max_text_bytes):
                continue
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                out += audit_text(rel, fh.read(), t, pool_labels_ok)
    return out


def bundle_files(out_dir):
    """Relative paths of everything in the directory, except NOT_BUNDLE names at its top level."""
    found = []
    for base, dirs, files in os.walk(out_dir):
        dirs.sort()
        top = os.path.abspath(base) == os.path.abspath(out_dir)
        for f in sorted(files):
            if not (top and f in NOT_BUNDLE):
                found.append(os.path.relpath(os.path.join(base, f), out_dir).replace(os.sep, "/"))
    return sorted(found)


def verify_bundle(out_dir, truth=None, rehash=True):
    """Check a bundle directory on disk. Returns (problems, manifest). An empty problem list means it is uploadable."""
    t = truth or load_truth()
    problems = list(split_problems(t))
    mpath = os.path.join(out_dir, "MANIFEST.json")
    if not os.path.isfile(mpath):
        return problems + [f"{mpath} is missing: run build_bundle.py first"], None
    with open(mpath) as fh:
        man = json.load(fh)
    listed = man.get("files", {})
    on_disk = set(bundle_files(out_dir)) - {"MANIFEST.json"}
    for extra in sorted(on_disk - set(listed))[:20]:
        problems.append(f"unexpected file in bundle (not in MANIFEST.json): {extra}")
    for missing in sorted(set(listed) - on_disk)[:20]:
        problems.append(f"file listed in MANIFEST.json is missing: {missing}")
    if rehash:
        for rel in sorted(set(listed) & on_disk):
            full = os.path.join(out_dir, rel)
            if os.path.getsize(full) != listed[rel]["bytes"] or sha256_file(full) != listed[rel]["sha256"]:
                problems.append(f"content differs from MANIFEST.json: {rel}")
    # structure
    tiles = {os.path.basename(r) for r in listed if r.startswith("gridded/")}
    allowed = t["pool"] | t["dev"]
    if tiles != allowed:
        problems.append(f"gridded/ holds {len(tiles)} tiles; expected exactly pool + dev = {len(allowed)} "
                        f"({len(tiles - allowed)} not allowed, {len(allowed - tiles)} missing)")
    in_test = sorted(tiles & t["test"])
    if in_test:
        problems.append(f"TEST-PARTITION TILES IN BUNDLE: {len(in_test)}, e.g. {in_test[:3]}")
    tm = os.path.join(out_dir, "train_manifest.csv")
    if os.path.isfile(tm):
        with open(tm, newline="") as fh:
            rows = list(csv.DictReader(fh))
        names = [r.get("image") for r in rows]
        if rows and list(rows[0].keys()) != ["image", "label", "ssa_votes"]:
            problems.append(f"train_manifest.csv columns are {list(rows[0].keys())}")
        if set(names) != t["pool"] or len(names) != len(t["pool"]):
            problems.append("train_manifest.csv does not list exactly the fewshot_pool tiles")
        if set(names) & t["dev"]:
            problems.append(f"DEV LABELS IN BUNDLE: train_manifest.csv lists {len(set(names) & t['dev'])} dev tiles")
        if set(names) & t["test"]:
            problems.append(f"TEST LABELS IN BUNDLE: train_manifest.csv lists {len(set(names) & t['test'])} test tiles")
        if any(t["label"].get(r["image"]) != r["label"] or str(t["votes"].get(r["image"])) != r["ssa_votes"] for r in rows
               if r.get("image") in t["label"]):
            problems.append("train_manifest.csv labels differ from annotations.csv")
    else:
        problems.append("train_manifest.csv is missing")
    dt = os.path.join(out_dir, "dev_tiles.csv")
    if os.path.isfile(dt):
        with open(dt, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows and list(rows[0].keys()) != ["image", "subset"]:
            problems.append(f"dev_tiles.csv columns are {list(rows[0].keys())}; only image,subset are allowed")
        if {r.get("image") for r in rows} != t["dev"]:
            problems.append("dev_tiles.csv does not list exactly the dev tiles")
    else:
        problems.append("dev_tiles.csv is missing")
    if f"prompts/{STEP4_PROMPT}" not in listed:
        problems.append(f"prompts/{STEP4_PROMPT} is missing")
    problems += audit_dir(out_dir, t)
    return problems, man


# ---------------------------------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------------------------------

def in_synced_folder(path):
    """True if `path` is somewhere macOS may sync to iCloud (Desktop & Documents, iCloud Drive)."""
    real = os.path.realpath(path)
    home = os.path.realpath(os.path.expanduser("~"))
    bad = [os.path.join(home, "Documents"), os.path.join(home, "Desktop"),
           os.path.join(home, "Library", "Mobile Documents"), os.path.join(home, "Library", "CloudStorage")]
    return any(real == b or real.startswith(b + os.sep) for b in bad)


def write_if_changed(path, data):
    """Atomic write; returns True if the file was (re)written, False if it already held these bytes."""
    if os.path.isfile(path) and os.path.getsize(path) == len(data):
        with open(path, "rb") as fh:
            if fh.read() == data:
                return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp-build"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)
    return True


def csv_bytes(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue().encode()


def prompt_sources():
    """{bundle name: source path} for every rendered prompt text (plus the rendered index)."""
    out = {}
    for d in PROMPT_DIRS:
        for f in sorted(os.listdir(d)):
            full = os.path.join(d, f)
            if os.path.isfile(full) and (f.endswith(".txt") or f == "index.json") and " " not in f:
                if f in out:
                    sys.exit(f"STOP: two prompt files are both called {f}")
                out[f] = full
    if STEP4_PROMPT not in out:
        sys.exit(f"STOP: {STEP4_PROMPT} not found under prompts/rendered")
    return out


def build(out_dir, prune=False, quiet=False):
    t = load_truth()
    problems = split_problems(t)
    if problems:
        sys.exit("STOP, nothing written:\n  " + "\n  ".join(problems))

    sys.path.insert(0, ROOT)
    import PIL
    from PIL import Image

    import grid
    import run_experiment as rx

    if not os.path.isfile(grid.FONT_PATH):
        sys.exit(f"STOP: {grid.FONT_PATH} is missing. grid.py would silently fall back to another font and the "
                 "tiles would differ from the pipeline's render.")

    tiles = sorted(t["pool"] | t["dev"])
    assert len(tiles) == EXPECT["fewshot_pool"] + EXPECT["dev"] and not set(tiles) & t["test"]
    assert all(t["partition"][n] == "train" for n in tiles)

    os.makedirs(out_dir, exist_ok=True)
    files, written = {}, 0

    def put(rel, data):
        nonlocal written
        written += write_if_changed(os.path.join(out_dir, rel), data)
        files[rel] = {"bytes": len(data), "sha256": sha256_bytes(data)}

    for i, name in enumerate(tiles, 1):
        assert name not in t["test"] and t["partition"][name] == "train", name       # never a test tile
        png = base64.b64decode(rx.b64_gridded_tile({"image": name, "gridded": None}))
        with Image.open(io.BytesIO(png)) as im:
            if im.size != (grid.TILE, grid.TILE) or im.format != "PNG":
                sys.exit(f"STOP: {name} rendered as {im.format} {im.size}")
        put(f"gridded/{name}", png)
        if not quiet and (i % 500 == 0 or i == len(tiles)):
            print(f"  rendered {i}/{len(tiles)} tiles", file=sys.stderr, flush=True)

    pool = sorted(t["pool"])
    assert not set(pool) & t["dev"] and not set(pool) & t["test"]                    # labels: pool tiles only
    put("train_manifest.csv", csv_bytes(["image", "label", "ssa_votes"],
                                        [[n, t["label"][n], t["votes"][n]] for n in pool]))
    screen = set(t["splits"]["screen"])
    put("dev_tiles.csv", csv_bytes(["image", "subset"],
                                   [[n, "screen" if n in screen else "dev_rest"] for n in sorted(t["dev"])]))
    prompts = prompt_sources()
    for f, src in prompts.items():
        with open(src, "rb") as fh:
            put(f"prompts/{f}", fh.read())
    put("NOTICE.txt", NOTICE.encode())

    n_hp = sum(t["label"][n] == "HP" for n in pool)
    manifest = {
        "bundle": "mhist-step4-private", "schema": SCHEMA, "private": True,
        "plan": "runs/competence/PLAN.md, Addendum: step 4",
        "counts": {"gridded_png": len(tiles), "fewshot_pool_tiles": len(pool), "dev_tiles": len(t["dev"]),
                   "dev_screen": len(screen), "dev_rest": len(t["dev"]) - len(screen),
                   "train_manifest_rows": len(pool), "train_manifest_HP": n_hp, "train_manifest_SSA": len(pool) - n_hp,
                   "prompt_files": len(prompts), "files_total": len(files),
                   "bytes_total": sum(v["bytes"] for v in files.values()),
                   "test_partition_tiles": 0, "dev_tiles_with_label": 0, "test_tiles_with_label": 0},
        "source": {"splits_json_sha256": sha256_file(SPLITS), "splits_seed": t["splits"].get("seed"),
                   "annotations_csv_sha256": sha256_file(ANNOTATIONS),
                   "step4_prompt": f"prompts/{STEP4_PROMPT}", "step4_prompt_sha256": files[f"prompts/{STEP4_PROMPT}"]["sha256"]},
        "render": {"function": "run_experiment.b64_gridded_tile -> grid.draw_grid", "pillow": PIL.__version__,
                   "font_file": grid.FONT_PATH, "font_index": grid.FONT_INDEX, "font_sha256": sha256_file(grid.FONT_PATH),
                   "tile_px": grid.TILE, "grid_n": grid.N, "grid_rgb": list(grid.GRID_RGB), "label_pt": grid.LABEL_PT},
        "files": dict(sorted(files.items())),
    }
    mbytes = (json.dumps(manifest, indent=1, sort_keys=False) + "\n").encode()
    written += write_if_changed(os.path.join(out_dir, "MANIFEST.json"), mbytes)

    stale = sorted(set(bundle_files(out_dir)) - set(files) - {"MANIFEST.json"})
    if stale and prune:
        for rel in stale:
            os.remove(os.path.join(out_dir, rel))
        print(f"  pruned {len(stale)} files that do not belong in the bundle", file=sys.stderr)
    problems, _ = verify_bundle(out_dir, t)
    if problems:
        print("BUNDLE NOT UPLOADABLE:", file=sys.stderr)
        for p in problems:
            print("  - " + p, file=sys.stderr)
        if stale and not prune:
            print("  (re-run with --prune to delete the unexpected files)", file=sys.stderr)
        sys.exit(2)
    return manifest, written, sha256_bytes(mbytes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"bundle directory (default: {DEFAULT_OUT}; env MHIST_BUNDLE_DIR)")
    ap.add_argument("--check", action="store_true", help="verify an existing bundle and exit; writes nothing")
    ap.add_argument("--prune", action="store_true", help="delete files in the bundle directory that are not part of the bundle")
    ap.add_argument("--allow-synced-dir", action="store_true",
                    help="allow --out inside Documents/Desktop/iCloud Drive (NOT recommended: tiles would sync to iCloud)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    out_dir = os.path.abspath(os.path.expanduser(args.out))
    if in_synced_folder(out_dir) and not args.allow_synced_dir:
        sys.exit(f"STOP: {out_dir} is inside a folder macOS may sync to iCloud. Choose a directory outside "
                 "Documents/Desktop/iCloud Drive (or pass --allow-synced-dir).")
    if args.check:
        problems, man = verify_bundle(out_dir)
        if problems:
            print(f"BUNDLE CHECK FAILED: {out_dir}", file=sys.stderr)
            for p in problems:
                print("  - " + p, file=sys.stderr)
            sys.exit(2)
        c = man["counts"]
        print(f"bundle OK: {out_dir}\n  {c['gridded_png']} gridded tiles ({c['fewshot_pool_tiles']} pool + {c['dev_tiles']} dev), "
              f"{c['train_manifest_rows']} labelled rows (pool only), {c['prompt_files']} prompt files, "
              f"{c['bytes_total'] / 1e6:.1f} MB\n  test-partition tiles: 0   dev tiles with a label: 0\n"
              f"  MANIFEST.json sha256 {sha256_file(os.path.join(out_dir, 'MANIFEST.json'))}")
        return
    man, written, msha = build(out_dir, prune=args.prune, quiet=args.quiet)
    c = man["counts"]
    print(f"bundle built: {out_dir}\n  {c['gridded_png']} gridded tiles ({c['fewshot_pool_tiles']} pool + {c['dev_tiles']} dev), "
          f"train_manifest.csv {c['train_manifest_rows']} rows (HP {c['train_manifest_HP']} / SSA {c['train_manifest_SSA']}, pool only), "
          f"{c['prompt_files']} prompt files, {c['files_total'] + 1} files, {c['bytes_total'] / 1e6:.1f} MB\n"
          f"  test-partition tiles: 0   dev tiles with a label: 0   files (re)written this run: {written}\n"
          f"  MANIFEST.json sha256 {msha}")


if __name__ == "__main__":
    main()
