#!/usr/bin/env python3
r"""Offline tests for the privacy code: kaggle_push.py and the leak tripwire in build_bundle.py.

    ~/mhist_local/venv/bin/python kaggle_ft/test_push_local.py            (about half a minute; add -v for test names)

NOTHING HERE TOUCHES THE NETWORK OR A KAGGLE ACCOUNT. kaggle_push.py is pointed at a stand-in for the `kaggle`
binary (--kaggle-bin): a small script written into the temp folder that keeps its "server" in a JSON file and can
be told to misbehave (report a dataset public, switch internet on, fail after the server has created something).
Credentials are a made-up kaggle.json in a temp KAGGLE_CONFIG_DIR; the project's .env is NOT opened (MHIST_DOTENV
points at a made-up file); verify-private always runs with --no-anon-probe.

Everything is written to a folder of this run's own under ~/mhist_local/tmp/ (MHIST_TEST_TMP names another base)
and removed at the end. The made-up "labels" in the evasion files are not the real ones. Fake credentials are
assembled at run time, so this file contains nothing a secret scanner would match.

Covered
    dry runs      every subcommand with --dry-run: exit 0, the stand-in is never started, nothing is written
    kinds         dataset uploads are allow-listed by kind (bundle / jobs / adapter / wheels); no kind = refused
    evasions      ten ways a dev label could ride along (indented JSON, 0/1 CSV, a bare label list, a line with
                  both labels, a CSV in a zip, an unknown extension, .ipynb, .gz, stems without MHIST_, an extra
                  JSONL field) are all refused; the audit rules for JSON documents, tables and archives
    bundle        MANIFEST-exact; a nested dataset-metadata.json / .DS_Store is an unexpected file
    staging       a folder inside a synced location is audited BEFORE it is copied to the staging area
    datasets      reported public / shared -> exit 3; processing error -> privacy checked first; a client failure
                  after the server created the dataset stays in the registry and is seen by verify-private
    notebooks     reported public -> exit 3; an existing public notebook is not pushed to; internet reported ON
                  although OFF was asked -> exit 3 and not recorded as confirmed; verify-private re-checks it;
                  internet ON needs --i-accept-internet
    verify        an unanswered metadata request never counts as private
    guard         --public / --update / delete cannot be run
    secrets       key formats (reported by family only), the kaggle.json key, KAGGLE_KEY, the access token and
                  .env values; --env names that look like credentials
    outcome       kernel-output prints train_summary.json / *.status.json status
"""

import contextlib
import io
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
import zipfile

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

TMP_BASE = os.path.abspath(os.path.expanduser(os.environ.get("MHIST_TEST_TMP") or "~/mhist_local/tmp"))
if os.path.realpath(TMP_BASE).startswith((os.path.realpath(ROOT) + os.sep, os.path.realpath(os.path.expanduser("~/Documents")) + os.sep)):
    raise SystemExit(f"the test folder base {TMP_BASE} is inside the project tree or Documents; set MHIST_TEST_TMP elsewhere")
os.makedirs(TMP_BASE, exist_ok=True)
TMP = tempfile.mkdtemp(prefix="mhist_test_push_", dir=TMP_BASE)
CFG = os.path.join(TMP, "kaggle_config")
FAKE_ENV = os.path.join(TMP, "fake_project.env")
SERVER = os.path.join(TMP, "fake_server.json")
STUB = os.path.join(TMP, "fake_kaggle")
OWNER = "tester-account"

# made-up credentials, assembled here so that no literal in this file looks like one
FAKE_KAGGLE_KEY = "0f" * 16
FAKE_ENV_VALUE = "fr" + "iendli-" + "x9" * 12
FAKE_ACCESS_TOKEN = "zq7" + "-" + "7q" * 14
FAKES = {"Hugging Face": "hf" + "_" + "Ab3" * 12, "Anthropic": "sk-" + "ant-" + "api03-" + "Zz9_" * 8,
         "OpenAI project": "sk-" + "proj-" + "Qw8-" * 8, "Google": "AI" + "za" + "Sy" + "D4" * 16 + "x",
         "Friendli": "fl" + "p_" + "M2" * 12, "OpenRouter": "sk-" + "or-v1-" + "a1" * 12,
         "Slack": "xo" + "xb-" + "1234567890-abcdef", "private key": "-----BEGIN " + "RSA PRIVATE KEY-----",
         "GitHub": "gh" + "p_" + "R5" * 12, "AWS": "AK" + "IA" + "B7" * 8}

os.environ["KAGGLE_CONFIG_DIR"] = CFG
os.environ["MHIST_DOTENV"] = FAKE_ENV               # the real project .env is never opened by these tests
os.environ["FAKE_KAGGLE_SERVER"] = SERVER
for _v in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN", "KAGGLE_PUSH_STARTED"):
    os.environ.pop(_v, None)

import build_bundle as bb      # noqa: E402
import kaggle_push as kp       # noqa: E402

STUB_SOURCE = r'''
import json, os, sys
path = os.environ["FAKE_KAGGLE_SERVER"]
st = json.load(open(path))
argv = sys.argv[1:]
st["calls"].append(argv)
knobs, owner = st["knobs"], st["owner"]
rc, out = 0, []

def opt(flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default

def uploaded(folder, mode):
    names = []
    for f in sorted(os.listdir(folder)):
        full = os.path.join(folder, f)
        if f == "dataset-metadata.json" or f == ".DS_Store" or f.startswith("._"):
            continue
        if os.path.isdir(full):
            if mode == "zip":
                names.append(f + ".zip")
        else:
            names.append(f)
    return names

cmd = tuple(argv[:2])
if cmd == ("config", "view"):
    out.append("Configuration values from " + os.environ.get("KAGGLE_CONFIG_DIR", "?") + "\n- username: " + owner + "\n- path: None")
elif argv[:1] == ["quota"]:
    out.append("resource  used   remaining  total\nGPU       1.00h  29.00h     30.00h")
elif cmd == ("datasets", "status"):
    d = st["datasets"].get(argv[2])
    if d is None:
        rc = 1
        out.append("404 Client Error: Not Found for url: https://www.kaggle.com/api/v1/datasets/status/" + argv[2])
    else:
        out.append(d["status"])
elif cmd in (("datasets", "create"), ("datasets", "version")):
    folder = opt("-p")
    meta = json.load(open(os.path.join(folder, "dataset-metadata.json")))
    ref = meta["id"]
    if cmd[1] == "create" and ref in st["datasets"]:
        rc = 1
        out.append("Dataset creation error: The requested title is already in use by a dataset")
    elif cmd[1] == "version" and ref not in st["datasets"]:
        rc = 1
        out.append("404 Client Error: Not Found")
    else:
        old = st["datasets"].get(ref, {})
        st["datasets"][ref] = {
            "private": old.get("private", not (knobs.get("create_public") or "-u" in argv or "--public" in argv)),
            "collaborators": old.get("collaborators", ["someone-else"] if knobs.get("create_shared") else []),
            "status": "error" if knobs.get("processing_error") else "ready",
            "files": uploaded(folder, opt("-r", "skip")), "versions": old.get("versions", 0) + 1}
        if knobs.get("client_fails_after_create"):
            rc = 1
            out.append("Dataset creation error: connection reset by peer")
        elif cmd[1] == "create":
            out.append("Your private Dataset is being created. Please check progress at https://www.kaggle.com/datasets/" + ref)
        else:
            out.append("Dataset version is being created. Please check progress at https://www.kaggle.com/datasets/" + ref)
elif cmd == ("datasets", "metadata"):
    ref, dest = argv[2], opt("-p")
    d = st["datasets"].get(ref)
    if ref in knobs.get("metadata_unknown", []):
        rc = 1
        out.append("500 Server Error: Internal Server Error")
    elif d is None:
        rc = 1
        out.append("404 Client Error: Not Found for url")
    else:
        info = {"title": ref.split("/")[1], "ownerUser": owner, "collaborators": [{"username": c, "role": "reader"} for c in d["collaborators"]]}
        if d["private"]:
            info["isPrivate"] = True
        os.makedirs(dest, exist_ok=True)
        json.dump(info, open(os.path.join(dest, "dataset-metadata.json"), "w"))
        out.append("Downloaded metadata to " + os.path.join(dest, "dataset-metadata.json"))
elif cmd == ("datasets", "list"):
    out.append("ref,title,size,lastUpdated,downloadCount,voteCount,usabilityRating")
    out += [r + ",t,1,2026-10-04,0,0,0.1" for r in sorted(st["datasets"]) if r not in knobs.get("unlisted", [])]
elif cmd == ("kernels", "push"):
    folder = opt("-p")
    meta = json.load(open(os.path.join(folder, "kernel-metadata.json")))
    ref = meta["id"]
    if knobs.get("push_error"):
        out.append("Kernel push error: Notebook not found")
    else:
        code = open(os.path.join(folder, meta["code_file"])).read()
        st["kernels"][ref] = {
            "is_private": False if knobs.get("push_public") else meta["is_private"] == "true",
            "enable_internet": True if knobs.get("internet_on") else meta["enable_internet"] == "true",
            "enable_gpu": meta["enable_gpu"] == "true", "machine_shape": meta.get("machine_shape"),
            "dataset_sources": meta["dataset_sources"], "kernel_sources": meta["kernel_sources"], "code": code,
            "versions": st["kernels"].get(ref, {}).get("versions", 0) + 1}
        out.append("Kernel version %d successfully pushed.  Please check progress at https://www.kaggle.com/code/%s" % (st["kernels"][ref]["versions"], ref))
elif cmd == ("kernels", "pull"):
    ref, dest = argv[2], opt("-p")
    k = st["kernels"].get(ref)
    if ref in knobs.get("metadata_unknown", []):
        rc = 1
        out.append("500 Server Error: Internal Server Error")
    elif k is None:
        rc = 1
        out.append("404 Client Error: Not Found for url")
    else:
        os.makedirs(dest, exist_ok=True)
        meta = {"id": ref, "title": ref.split("/")[1], "code_file": "x.py", "language": "python", "kernel_type": "script",
                "is_private": k["is_private"], "enable_gpu": k["enable_gpu"], "enable_internet": k["enable_internet"],
                "dataset_sources": k["dataset_sources"], "kernel_sources": k["kernel_sources"], "machine_shape": k["machine_shape"]}
        if knobs.get("internet_unreported"):
            del meta["enable_internet"]
        json.dump(meta, open(os.path.join(dest, "kernel-metadata.json"), "w"))
        out.append("Source code and metadata downloaded to " + dest)
elif cmd == ("kernels", "status"):
    out.append('%s has status "KernelWorkerStatus.COMPLETE"' % argv[2])
elif cmd == ("kernels", "output"):
    dest = opt("-p")
    for rel, content in knobs.get("output_files", {}).items():
        os.makedirs(os.path.dirname(os.path.join(dest, rel)) or dest, exist_ok=True)
        open(os.path.join(dest, rel), "w").write(content)
        out.append("Output file downloaded to " + os.path.join(dest, rel))
elif cmd == ("kernels", "list"):
    out.append("ref,title,author,lastRunTime,totalVotes")
    out += [r + ",t,a,2026-10-04,0" for r in sorted(st["kernels"])]
else:
    rc = 2
    out.append("unknown command: " + " ".join(argv))
json.dump(st, open(path, "w"))
print("\n".join(out))
sys.exit(rc)
'''


def server(**knobs):
    """Reset the stand-in's server. Returns nothing; read it back with state()."""
    with open(SERVER, "w") as fh:
        json.dump({"owner": OWNER, "datasets": {}, "kernels": {}, "calls": [], "knobs": knobs}, fh)


def state():
    with open(SERVER) as fh:
        return json.load(fh)


def set_state(fn):
    st = state()
    fn(st)
    with open(SERVER, "w") as fh:
        json.dump(st, fh)


def calls(*prefix):
    return [c for c in state()["calls"] if tuple(c[:len(prefix)]) == prefix]


def run(*argv, state_dir=None, dry=False):
    """kaggle_push.main in process. Returns (exit code, stdout + stderr)."""
    kp._OWN_SECRETS = None
    buf = io.StringIO()
    full = [argv[0], *argv[1:], "--kaggle-bin", STUB, "--state-dir", state_dir or os.path.join(TMP, "state"),
            "--retry-seconds", "0", "--poll-seconds", "0"] + (["--dry-run"] if dry else [])
    code = 0
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            kp.main(full)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return code, buf.getvalue()


def registry(state_dir=None):
    path = os.path.join(state_dir or os.path.join(TMP, "state"), "registry.json")
    if not os.path.exists(path):
        return {"datasets": {}, "kernels": {}}
    with open(path) as fh:
        return json.load(fh)


def fresh_state_dir(name):
    d = os.path.join(TMP, "state_" + name)
    shutil.rmtree(d, ignore_errors=True)
    return d


TRUTH = bb.load_truth()
DEV = sorted(TRUTH["dev"])
POOL = sorted(TRUTH["pool"])
TEST_TILE = sorted(TRUTH["test"])[0]
PROMPT = open(os.path.join(ROOT, "prompts", "rendered", "cte_p1.txt")).read()


def job(tile, tiles="dev", **extra):
    stem = tile[:-4]
    return {"job_id": f"{tiles}__cte_p1__none__tier1__{stem}", "tier": "1", "seed": 7, "max_new_tokens": 1500,
            "parts": [{"type": "image", "file": f"gridded/{tile}", "sha256": "0" * 64}, {"type": "text", "text": PROMPT}], **extra}


def make_dir(name, files):
    """A folder under TMP holding {relative path: str | bytes}."""
    d = os.path.join(TMP, name)
    shutil.rmtree(d, ignore_errors=True)
    for rel, content in files.items():
        full = os.path.join(d, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as fh:
            fh.write(content if isinstance(content, bytes) else content.encode())
    os.makedirs(d, exist_ok=True)
    return d


def jobs_dir(name="jobs_ok", extra=None):
    files = {"dev__cte_p1__none__tier1.jsonl": "".join(json.dumps(job(t)) + "\n" for t in DEV[:5])}
    files.update(extra or {})
    return make_dir(name, files)


def adapter_dir(name="adapter_ok", extra=None, drop=()):
    files = {"adapter_config.json": json.dumps({"r": 16, "peft_type": "LORA"}), "adapter_model.safetensors": b"\x00weights\x01",
             "step4_meta.json": json.dumps({"epoch": 3, "final": True, "protocol_run": True, "target_ids": {"HP": [1], "SSA": [2]}}),
             "README.md": "adapter card\n"}
    files.update(extra or {})
    return make_dir(name, {k: v for k, v in files.items() if k not in drop})


def zip_bytes(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, c in members.items():
            z.writestr(n, c)
    return buf.getvalue()


def made_up_label(i):
    return "SSA" if i % 3 == 0 else "HP"               # NOT the real labels


def setUpModule():
    os.makedirs(CFG)
    with open(os.path.join(CFG, "kaggle.json"), "w") as fh:
        json.dump({"username": OWNER, "key": FAKE_KAGGLE_KEY}, fh)
    os.chmod(os.path.join(CFG, "kaggle.json"), 0o600)
    with open(FAKE_ENV, "w") as fh:
        fh.write(f"# made-up\nexport FRIENDLI_TOKEN='{FAKE_ENV_VALUE}'\nSHORT=abc\n")
    with open(STUB, "w") as fh:
        fh.write("#!" + sys.executable + "\n" + STUB_SOURCE)
    os.chmod(STUB, os.stat(STUB).st_mode | stat.S_IXUSR)
    server()


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)             # this run's own folder only


class DryRuns(unittest.TestCase):
    def test_every_subcommand_dry(self):
        server()
        sd = fresh_state_dir("dry")
        script = os.path.join(HERE, "train_lora.py")
        cases = [("check",), ("quota",), ("verify-private", "--no-anon-probe"), ("dataset-status", "--slug", "mhist-priv-bundle"),
                 ("dataset-create", "--kind", "jobs", "--dir", jobs_dir()),
                 ("dataset-create", "--kind", "adapter", "--dir", adapter_dir()),
                 ("dataset-version", "--kind", "jobs", "--dir", jobs_dir(), "-m", "v2"),
                 ("kernel-push", "--script", script, "--slug", "mhist-priv-probe", "--dataset", "bundle", "--dataset", "weights",
                  "--args=--verify-only --model-dir {weights} --data-dir {bundle}"),
                 ("kernel-push", "--script", os.path.join(HERE, "infer_jobs.py"), "--slug", "mhist-priv-infer-none-t1", "--dataset", "bundle",
                  "--dataset", "weights", "--dataset", "jobs", "--dataset", "adapter", "--dataset", "wheels",
                  "--pip-install", "peft==0.0.0", "--pip-find-links", "{wheels}",
                  "--args=--jobs {jobs}/dev__cte_p1__none__tier1.jsonl --bundle-dir {bundle} --model-dir {weights} --adapter-dir {adapter}"),
                 ("kernel-status", "--slug", "train", "--wait"), ("kernel-output", "--slug", "train")]
        if os.path.isfile(os.path.join(bb.DEFAULT_OUT, "MANIFEST.json")):
            cases += [("dataset-create",), ("dataset-version", "-m", "v2")]
        if os.path.isfile(os.path.join(kp.WEIGHTS_DIR, "config.json")):
            cases.append(("weights",))
        for argv in cases:
            code, out = run(*argv, state_dir=sd, dry=True)
            self.assertEqual(code, 0, f"{argv[0]}: exit {code}\n{out[-600:]}")
            self.assertIn("[dry-run]", out)
        self.assertEqual(state()["calls"], [], "a dry run started the kaggle binary")
        self.assertFalse(os.path.exists(sd), "a dry run wrote to the state directory")
        code, out = run("kernel-push", "--script", script, "--slug", "mhist-priv-probe", "--args=--verify-only", state_dir=sd, dry=True)
        self.assertIn("_kp_sys.argv = ['train_lora.py'] + ['--verify-only']", out)       # argv[0] is replaced too
        self.assertIn('"enable_internet": "false"', out)
        self.assertIn('"is_private": "true"', out)


class UploadKinds(unittest.TestCase):
    def refused(self, argv, needle=None, code=kp.EXIT_LEAK):
        server()
        got, out = run(*argv, state_dir=fresh_state_dir("kinds"))
        self.assertEqual(got, code, f"{argv}: exit {got}\n{out[-800:]}")
        self.assertEqual(calls("datasets", "create") + calls("datasets", "version"), [], f"{argv}: something was uploaded")
        if needle:
            self.assertIn(needle, out)
        return out

    def test_kind_is_required_and_must_fit(self):
        self.refused(("dataset-create", "--dir", jobs_dir()), "--kind is required", kp.EXIT_USAGE)
        self.refused(("dataset-create", "--kind", "bundle", "--dir", jobs_dir()), "does not fit", kp.EXIT_USAGE)
        self.refused(("dataset-create", "--kind", "adapter", "--dir", jobs_dir()), "not a file this kind of upload may hold")
        self.refused(("dataset-create", "--kind", "wheels", "--dir", adapter_dir()), "not a file this kind of upload may hold")
        self.refused(("dataset-create", "--kind", "jobs", "--dir", adapter_dir()), "not a file this kind of upload may hold")
        with self.assertRaises(SystemExit):                    # there is no kind for arbitrary data
            with contextlib.redirect_stderr(io.StringIO()):
                kp.main(["dataset-create", "--kind", "data", "--dir", jobs_dir(), "--dry-run"])

    def test_happy_paths_upload_only_the_kind(self):
        sd = fresh_state_dir("happy")
        server()
        for kind, folder, slug in (("jobs", jobs_dir(), "mhist-priv-jobs"), ("adapter", adapter_dir(), "mhist-priv-adapter"),
                                   ("wheels", make_dir("wheels_ok", {"peft-1.0-py3-none-any.whl": zip_bytes({"peft/__init__.py": "x = 1\n"})}),
                                    "mhist-priv-wheels")):
            code, out = run("dataset-create", "--kind", kind, "--dir", folder, state_dir=sd)
            self.assertEqual(code, 0, out[-800:])
            ref = f"{OWNER}/{slug}"
            self.assertIn(f"PRIVATE confirmed by Kaggle: {ref}", out)
            entry = registry(sd)["datasets"][ref]
            self.assertEqual((entry["kind"], entry["status"]), (kind, "ready"))
            self.assertTrue(entry["private_confirmed"])
            self.assertNotIn("dataset-metadata.json", state()["datasets"][ref]["files"])
            create = [c for c in calls("datasets", "create") if c[c.index("-p") + 1] == folder][0]
            self.assertEqual(create[create.index("-r") + 1], "skip")           # flat kinds: no sub-directory is ever uploaded
            self.assertFalse(kp.FORBIDDEN_FLAGS & set(create))
        code, out = run("verify-private", "--no-anon-probe", state_dir=sd)
        self.assertEqual(code, 0, out[-600:])
        self.assertIn("all 3 private", out)
        # the same content again: nothing is uploaded twice
        n = len(calls("datasets", "create"))
        code, out = run("dataset-create", "--kind", "jobs", "--dir", jobs_dir(), state_dir=sd)
        self.assertEqual((code, len(calls("datasets", "create"))), (0, n), out[-400:])
        self.assertIn("not uploading again", out)

    def test_ten_evasions_are_refused(self):
        dev = DEV[:40]
        indented = json.dumps([{"image": t, "meta": {"answer": made_up_label(i)}} for i, t in enumerate(dev)], indent=2)
        evasions = {
            "indented JSON, name and label on different lines": {"labels.json": indented},
            "CSV with 0/1 codes": {"codes.csv": "image,y\n" + "".join(f"{t},{i % 2}\n" for i, t in enumerate(dev))},
            "a bare label list in dev order": {"order.txt": "\n".join(made_up_label(i) for i in range(300)) + "\n"},
            "a line with both labels": {"both.txt": "".join(f"{t}: {made_up_label(i)} (not {'HP' if made_up_label(i) == 'SSA' else 'SSA'})\n"
                                                             for i, t in enumerate(dev))},
            "a labels CSV inside a zip": {"inner.zip": zip_bytes({"labels.csv": "image,label\n" + "".join(f"{t},{made_up_label(i)}\n" for i, t in enumerate(dev))})},
            "an unknown extension": {"labels.dat": "".join(f"{t}\t{made_up_label(i)}\n" for i, t in enumerate(dev))},
            "a notebook": {"labels.ipynb": json.dumps({"cells": [{"source": [f"{t} {made_up_label(i)}" for i, t in enumerate(dev)]}]})},
            "a gzip file": {"labels.csv.gz": b"\x1f\x8b\x08\x00" + b"\x00" * 20},
            "tile stems without the MHIST_ prefix": {"stems.csv": "id,y\n" + "".join(f"{t[6:9]},{made_up_label(i)}\n" for i, t in enumerate(dev))},
            "an extra JSONL field next to the prompt": {"extra.jsonl": "".join(json.dumps(job(t, y=made_up_label(i))) + "\n" for i, t in enumerate(dev))},
        }
        self.assertEqual(len(evasions), 10)
        for what, files in evasions.items():
            for kind, folder in (("jobs", jobs_dir("evasion_jobs", files)), ("adapter", adapter_dir("evasion_adapter", files)),
                                 ("wheels", make_dir("evasion_wheels", {"a-1-py3-none-any.whl": zip_bytes({"a.py": "x=1"}), **files}))):
                out = self.refused(("dataset-create", "--kind", kind, "--dir", folder))
                self.assertIn("LEAK TRIPWIRE", out, f"{what} / {kind}")
        # the label-bearing evasions hidden in an allow-listed NAME of the adapter kind are caught by content
        for what, name, content in (("indented JSON", "step4_meta.json", indented),
                                    ("label keys", "adapter_config.json", json.dumps({"tiles": dev[:3], "n_ssa": 2}, indent=1))):
            out = self.refused(("dataset-create", "--kind", "adapter", "--dir", adapter_dir("evasion_named", {name: content})))
            self.assertIn("the document names dev tiles", out, what)

    def test_structure_rules(self):
        t = DEV[0]
        for what, folder, needle in (
                ("a sidecar in a jobs folder", jobs_dir("j1", {"dev__cte_p1__none__tier1.sidecar.jsonl": json.dumps({"job_id": "x"}) + "\n"}), "must never be uploaded"),
                ("a meta file", jobs_dir("j2", {"dev__cte_p1__none__tier1.meta.json": "{}"}), "must never be uploaded"),
                ("the DO_NOT_UPLOAD marker", jobs_dir("j3", {"DO_NOT_UPLOAD.txt": "x"}), "must never be uploaded"),
                ("a local_only sub-folder", jobs_dir("j4", {"local_only/x.jsonl": json.dumps(job(t)) + "\n"}), "local_only"),
                ("a sub-directory", jobs_dir("j5", {"more/x.jsonl": json.dumps(job(t)) + "\n"}), "flat folder"),
                ("a job with a label key", jobs_dir("j6", {"x.jsonl": json.dumps(job(t, label_true="HP")) + "\n"}), "not a job file"),
                ("a job whose image is outside gridded/", jobs_dir("j7", {"x.jsonl": json.dumps({**job(t), "parts": [{"type": "image", "file": "other/" + t}]}) + "\n"}), "not gridded/MHIST_xxx.png"),
                ("a job naming a test tile", jobs_dir("j8", {"x.jsonl": json.dumps(job(TEST_TILE)) + "\n"}), "test-partition tile"),
                ("a line that is not JSON", jobs_dir("j9", {"x.jsonl": "MHIST not json\n"}), "not a job file"),
                ("a nested dataset-metadata.json", jobs_dir("j10", {"sub/dataset-metadata.json": "{}"}), "not a file this kind"),
                ("a nested .DS_Store", jobs_dir("j11", {"sub/.DS_Store": b"\x00\x01"}), "not a file this kind")):
            out = self.refused(("dataset-create", "--kind", "jobs", "--dir", folder), needle)
            self.assertIn("LEAK TRIPWIRE", out, what)
        self.refused(("dataset-create", "--kind", "jobs", "--dir", make_dir("j_empty", {})), "no job file")
        self.refused(("dataset-create", "--kind", "adapter", "--dir", adapter_dir("a1", drop=("step4_meta.json",))), "step4_meta.json is missing")
        self.refused(("dataset-create", "--kind", "adapter", "--dir", adapter_dir("a2", {"optimizer.pt": b"\x00"})), "not a file this kind")
        self.refused(("dataset-create", "--kind", "adapter", "--dir", adapter_dir("a3", {"adapter_config.json": "[1, 2]"})), "JSON object")
        self.refused(("dataset-create", "--kind", "wheels", "--dir", make_dir("w1", {"x-1-py3-none-any.whl": zip_bytes(
            {"x/data/" + TEST_TILE: "png"})})), "test-partition tile")
        self.refused(("dataset-create", "--kind", "wheels", "--dir", make_dir("w2", {"x-1-py3-none-any.whl": zip_bytes(
            {"x/dev.sidecar.jsonl": "{}"})})), "must never be uploaded")
        self.refused(("dataset-create", "--kind", "wheels", "--dir", make_dir("w3", {"x-1-py3-none-any.whl": b"not a zip"})), "unreadable zip")
        # a top-level dataset-metadata.json / .DS_Store is the one exemption (the Kaggle CLI does not upload them from there)
        server()
        code, out = run("dataset-create", "--kind", "jobs", "--dir", jobs_dir("j_ok", {".DS_Store": b"\x00\x01", "dataset-metadata.json": "{}"}),
                        state_dir=fresh_state_dir("kinds_ok"))
        self.assertEqual(code, 0, out[-600:])
        self.assertEqual(state()["datasets"][f"{OWNER}/mhist-priv-jobs"]["files"], ["dev__cte_p1__none__tier1.jsonl"])

    def test_audit_rules_directly(self):
        dev, pool = DEV[:3], POOL[:3]
        flag = lambda rel, text: bb.audit_text(rel, text, TRUTH)                                   # noqa: E731
        self.assertEqual(flag("dev_tiles.csv", "image,subset\n" + "".join(f"{t},screen\n" for t in dev)), [])
        self.assertTrue(flag("dev_tiles.csv", "image,subset\n" + f"{dev[0]},1\n"))                # a code where the subset belongs
        self.assertTrue(flag("x.csv", "image,subset,extra\n" + f"{dev[0]},screen,0\n"))
        self.assertTrue(flag("x.tsv", "image\tsubset\n" + f"{dev[0]}\tscreen\t1\n"))
        self.assertEqual(flag("train_manifest.csv", "image,label,ssa_votes\n" + "".join(f"{t},HP,0\n" for t in pool)), [])   # pool labels are allowed
        self.assertEqual(flag("names.json", json.dumps({"files": dev}, indent=1)), [])            # names alone are fine (MANIFEST.json)
        self.assertTrue(flag("x.json", json.dumps({"files": dev, "note": "SSA"}, indent=1)))
        self.assertTrue(flag("x.json", json.dumps({"files": dev, "ssa_votes": [1, 2, 3]}, indent=1)))
        self.assertEqual(flag("x.json", json.dumps({"files": pool, "note": "SSA"}, indent=1)), [])
        self.assertTrue(flag("x.txt", TEST_TILE))                                                  # a test tile anywhere
        self.assertEqual(flag("jobs.jsonl", json.dumps(job(dev[0]))), [])                         # the prompt names both classes
        self.assertTrue(flag("jobs.jsonl", json.dumps({**job(dev[0]), "y": "SSA"}).replace(json.dumps(PROMPT), '"p"')))
        self.assertIsNotNone(bb.forbidden_path("a/local_only/x.txt"))
        self.assertIsNotNone(bb.forbidden_path("x.sidecar.jsonl"))
        self.assertIsNone(bb.forbidden_path("step4_meta.json"))
        d = make_dir("audit_zip", {"a.zip": zip_bytes({"deep/labels.csv": "image,label\n" + f"{dev[0]},HP\n"})})
        self.assertTrue(any("a.zip!deep/labels.csv" in p for p in bb.audit_dir(d, TRUTH)))


@unittest.skipUnless(os.path.isfile(os.path.join(bb.DEFAULT_OUT, "MANIFEST.json")), "the data bundle is not built on this machine")
class Bundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.copy = os.path.join(TMP, "bundle_copy")        # hard links: no extra disk; the tests only ADD files to it
        shutil.copytree(bb.DEFAULT_OUT, cls.copy, copy_function=os.link, ignore=shutil.ignore_patterns("dataset-metadata.json", ".DS_Store"))

    def test_bundle_is_manifest_exact(self):
        problems, man = bb.verify_bundle(self.copy, TRUTH)
        self.assertEqual(problems, [])
        self.assertEqual(man["counts"]["dev_tiles_with_label"], 0)
        for rel in ("gridded/dataset-metadata.json", "gridded/.DS_Store", "prompts/.DS_Store", "extra.txt"):
            full = os.path.join(self.copy, rel)
            with open(full, "w") as fh:
                fh.write("{}")
            try:
                problems, _ = bb.verify_bundle(self.copy, TRUTH, rehash=False)
                self.assertTrue(any("unexpected file" in p and rel in p for p in problems), f"{rel} deeper down was not reported: {problems[:2]}")
                server()
                code, out = run("dataset-create", "--dir", self.copy, state_dir=fresh_state_dir("bundle"))
                self.assertEqual((code, calls("datasets", "create")), (kp.EXIT_LEAK, []), out[-400:])
            finally:
                os.remove(full)
        for rel in ("dataset-metadata.json", ".DS_Store"):     # at the top level they are not part of the upload
            with open(os.path.join(self.copy, rel), "w") as fh:
                fh.write("{}")
        try:
            self.assertEqual(bb.verify_bundle(self.copy, TRUTH, rehash=False)[0], [])
        finally:
            for rel in ("dataset-metadata.json", ".DS_Store"):
                os.remove(os.path.join(self.copy, rel))

    def test_bundle_upload_and_registry(self):
        sd = fresh_state_dir("bundle_up")
        server()
        code, out = run("dataset-create", "--dir", self.copy, state_dir=sd)
        self.assertEqual(code, 0, out[-800:])
        ref = f"{OWNER}/mhist-priv-bundle"
        files = state()["datasets"][ref]["files"]
        self.assertEqual(sorted(files), sorted(["MANIFEST.json", "NOTICE.txt", "dev_tiles.csv", "train_manifest.csv", "gridded.zip", "prompts.zip"]))
        self.assertEqual(registry(sd)["datasets"][ref]["kind"], "bundle")
        os.remove(os.path.join(self.copy, "dataset-metadata.json"))


class Staging(unittest.TestCase):
    def test_audit_runs_before_anything_is_copied(self):
        """A folder inside a synced location is mirrored to the staging area: only after the tripwire has passed."""
        real = bb.in_synced_folder
        bad = jobs_dir("synced_bad", {"dev__cte_p1__none__tier1.sidecar.jsonl": json.dumps({"job_id": "x", "label_true": "HP"}) + "\n"})
        good = jobs_dir("synced_good")
        bb.in_synced_folder = lambda p: os.path.realpath(p) in (os.path.realpath(bad), os.path.realpath(good)) or real(p)
        try:
            sd = fresh_state_dir("staging")
            server()
            code, out = run("dataset-create", "--kind", "jobs", "--dir", bad, state_dir=sd)
            self.assertEqual(code, kp.EXIT_LEAK, out[-400:])
            self.assertFalse(os.path.exists(os.path.join(sd, "datasets")), "the folder was copied to the staging area before the audit")
            code, out = run("dataset-create", "--kind", "jobs", "--dir", good, state_dir=sd)
            self.assertEqual(code, 0, out[-600:])
            stage = os.path.join(sd, "datasets", "mhist-priv-jobs")
            self.assertEqual(calls("datasets", "create")[0][3], stage)        # uploaded from the staged copy
            self.assertFalse(os.path.exists(os.path.join(good, "dataset-metadata.json")))   # nothing written into the source
        finally:
            bb.in_synced_folder = real


class DatasetPrivacy(unittest.TestCase):
    def create(self, name, **knobs):
        sd = fresh_state_dir(name)
        server(**knobs)
        code, out = run("dataset-create", "--kind", "jobs", "--dir", jobs_dir(), state_dir=sd)
        return sd, code, out, registry(sd)["datasets"].get(f"{OWNER}/mhist-priv-jobs")

    def test_reported_public(self):
        sd, code, out, entry = self.create("public", create_public=True)
        self.assertEqual(code, kp.EXIT_PRIVACY, out[-600:])
        self.assertIn("PRIVACY CHECK FAILED - ACT NOW", out)
        self.assertIsNotNone(entry, "the dataset is not in the registry")
        self.assertIsNone(entry["private_confirmed"])
        code, out = run("verify-private", "--no-anon-probe", state_dir=sd)
        self.assertEqual(code, kp.EXIT_PRIVACY)
        self.assertIn("FAIL  dataset", out)

    def test_reported_shared(self):
        sd, code, out, entry = self.create("shared", create_shared=True)
        self.assertEqual(code, kp.EXIT_PRIVACY, out[-600:])
        self.assertIn("collaborator", out)
        self.assertIsNone(entry["private_confirmed"])

    def test_processing_error_after_create(self):
        sd, code, out, entry = self.create("procerr", processing_error=True)
        self.assertEqual(code, kp.EXIT_USAGE, out[-600:])
        self.assertIn("error while processing", out)
        self.assertIn("PRIVATE confirmed by Kaggle", out)                      # privacy was checked before the error was reported
        self.assertTrue(entry["private_confirmed"])
        self.assertEqual(entry["status"], "processing_error")
        self.assertEqual(run("verify-private", "--no-anon-probe", state_dir=sd)[0], 0)

    def test_processing_error_and_public(self):
        sd, code, out, entry = self.create("procerr_public", processing_error=True, create_public=True)
        self.assertEqual(code, kp.EXIT_PRIVACY, out[-600:])                    # the privacy alarm wins over the processing error
        self.assertIsNone(entry["private_confirmed"])

    def test_client_failure_after_server_create(self):
        sd = fresh_state_dir("clientfail")
        server(client_fails_after_create=True, create_public=True)
        code, out = run("dataset-create", "--kind", "jobs", "--dir", jobs_dir(), "--slug", "odd-slug-1", "--any-slug", state_dir=sd)
        self.assertEqual(code, kp.EXIT_PRIVACY, out[-600:])                    # the server has it, and it is public: alarm, not a plain failure
        ref = f"{OWNER}/odd-slug-1"
        self.assertEqual(registry(sd)["datasets"][ref]["status"], "upload_failed")
        code, out = run("verify-private", "--no-anon-probe", state_dir=sd)     # found through the registry although the slug has no prefix
        self.assertEqual(code, kp.EXIT_PRIVACY)
        self.assertIn(ref, out)
        # the same failure with a private result is a plain failure, still registered
        sd = fresh_state_dir("clientfail2")
        server(client_fails_after_create=True)
        code, out = run("dataset-create", "--kind", "jobs", "--dir", jobs_dir(), "--slug", "odd-slug-2", "--any-slug", state_dir=sd)
        self.assertEqual(code, kp.EXIT_USAGE, out[-600:])
        self.assertIn(f"{OWNER}/odd-slug-2", registry(sd)["datasets"])
        self.assertEqual(run("verify-private", "--no-anon-probe", state_dir=sd)[0], 0)

    def test_unknown_metadata_is_never_private(self):
        sd, code, out, entry = self.create("unknown")
        self.assertEqual(code, 0, out[-600:])
        ref = f"{OWNER}/mhist-priv-jobs"
        set_state(lambda st: st["knobs"].update(metadata_unknown=[ref], unlisted=[ref]))
        code, out = run("verify-private", "--no-anon-probe", state_dir=sd)     # Kaggle does not answer, and the listing lacks it
        self.assertEqual(code, kp.EXIT_PRIVACY, out[-600:])
        self.assertIn("unknown", out)
        set_state(lambda st: (st["knobs"].update(metadata_unknown=[]), st["datasets"].pop(ref)))
        code, out = run("verify-private", "--no-anon-probe", state_dir=sd)     # a clear "does not exist" + not listed = gone
        self.assertEqual(code, 0, out[-600:])
        self.assertIn("gone", out)

    def test_version_of_a_public_dataset_is_not_uploaded(self):
        sd, code, out, entry = self.create("version")
        self.assertEqual(code, 0)
        set_state(lambda st: st["datasets"][f"{OWNER}/mhist-priv-jobs"].update(private=False))
        code, out = run("dataset-version", "--kind", "jobs", "--dir", jobs_dir("jobs_v2", {"x.jsonl": json.dumps(job(DEV[9])) + "\n"}),
                        "-m", "v2", state_dir=sd)
        self.assertEqual((code, calls("datasets", "version")), (kp.EXIT_PRIVACY, []), out[-400:])


class NotebookPrivacy(unittest.TestCase):
    script = None

    @classmethod
    def setUpClass(cls):
        cls.script = os.path.join(TMP, "probe_script.py")
        with open(cls.script, "w") as fh:
            fh.write('"""A harmless script."""\nimport sys\nprint(sys.argv)\n')

    def push(self, name, *extra, prepare=None, **knobs):
        sd = fresh_state_dir(name)
        server(**knobs)
        if prepare:
            set_state(prepare)
        code, out = run("kernel-push", "--script", self.script, "--slug", "mhist-priv-probe", "--args=--verify-only", *extra, state_dir=sd)
        return sd, code, out, registry(sd)["kernels"].get(f"{OWNER}/mhist-priv-probe")

    def test_happy_push(self):
        sd, code, out, entry = self.push("nb_ok")
        self.assertEqual(code, 0, out[-800:])
        self.assertIn("PRIVATE confirmed by Kaggle", out)
        self.assertTrue(entry["private_confirmed"])
        k = state()["kernels"][f"{OWNER}/mhist-priv-probe"]
        self.assertEqual((k["is_private"], k["enable_internet"]), (True, False))
        self.assertIn("_kp_sys.argv = ['probe_script.py'] + ['--verify-only']", k["code"])
        self.assertEqual(run("verify-private", "--no-anon-probe", state_dir=sd)[0], 0)

    def test_reported_public_after_push(self):
        sd, code, out, entry = self.push("nb_public", push_public=True)
        self.assertEqual(code, kp.EXIT_PRIVACY, out[-600:])
        self.assertIsNone(entry["private_confirmed"])
        self.assertEqual(run("verify-private", "--no-anon-probe", state_dir=sd)[0], kp.EXIT_PRIVACY)

    def test_existing_public_notebook_is_not_pushed_to(self):
        ref = f"{OWNER}/mhist-priv-probe"
        existing = {"is_private": False, "enable_internet": False, "enable_gpu": True, "machine_shape": None,
                    "dataset_sources": [], "kernel_sources": [], "code": "", "versions": 1}
        sd, code, out, entry = self.push("nb_existing_public", prepare=lambda st: st["kernels"].update({ref: existing}))
        self.assertEqual(code, kp.EXIT_PRIVACY, out[-600:])
        self.assertEqual(calls("kernels", "push"), [], "the script was pushed to a notebook that is public")
        self.assertIn("nothing was pushed", out)
        # exists, but Kaggle does not answer the metadata request: not pushed either
        sd, code, out, entry = self.push("nb_existing_unknown", prepare=lambda st: (st["kernels"].update({ref: dict(existing, is_private=True)}),
                                                                                    st["knobs"].update(metadata_unknown=[ref])))
        self.assertEqual((code, calls("kernels", "push")), (kp.EXIT_USAGE, []), out[-400:])

    def test_internet_reported_on_is_a_failure(self):
        sd, code, out, entry = self.push("nb_internet", internet_on=True)
        self.assertEqual(code, kp.EXIT_PRIVACY, out[-800:])
        self.assertIn("internet ON", out)
        self.assertIn("stop the run on kaggle.com now", out)
        self.assertNotIn("PRIVATE confirmed by Kaggle", out)
        self.assertIsNone(entry["private_confirmed"])
        self.assertIn("ALARM", entry["status"])
        code, out = run("verify-private", "--no-anon-probe", state_dir=sd)
        self.assertEqual(code, kp.EXIT_PRIVACY, out[-600:])
        self.assertIn("internet ON, expected OFF", out)
        set_state(lambda st: st["kernels"][f"{OWNER}/mhist-priv-probe"].update(enable_internet=False))
        self.assertEqual(run("verify-private", "--no-anon-probe", state_dir=sd)[0], 0)

    def test_internet_setting_not_reported_is_not_taken_on_trust(self):
        sd, code, out, entry = self.push("nb_internet_unreported", internet_unreported=True)
        self.assertEqual(code, kp.EXIT_PRIVACY, out[-600:])
        self.assertIn("did not report the internet setting", out)
        self.assertIsNone(entry["private_confirmed"])
        self.assertEqual(run("verify-private", "--no-anon-probe", state_dir=sd)[0], kp.EXIT_PRIVACY)

    def test_internet_needs_explicit_acceptance(self):
        sd, code, out, entry = self.push("nb_net_flag", "--enable-internet", "true", "--pip-install", "peft")
        self.assertEqual((code, calls("kernels", "push")), (kp.EXIT_USAGE, []), out[-400:])
        self.assertIn("--i-accept-internet", out)
        sd, code, out, entry = self.push("nb_net_ok", "--enable-internet", "true", "--i-accept-internet", "--pip-install", "peft")
        self.assertEqual(code, 0, out[-600:])
        self.assertTrue(entry["enable_internet"])
        self.assertEqual(run("verify-private", "--no-anon-probe", state_dir=sd)[0], 0)    # pushed that way: not a mismatch

    def test_public_or_shared_sources_are_not_attached(self):
        sd = fresh_state_dir("nb_sources")
        server()
        ref = f"{OWNER}/mhist-priv-jobs"
        self.assertEqual(run("dataset-create", "--kind", "jobs", "--dir", jobs_dir(), state_dir=sd)[0], 0)
        set_state(lambda st: st["datasets"][ref].update(private=False))
        code, out = run("kernel-push", "--script", self.script, "--slug", "mhist-priv-probe", "--dataset", "jobs", state_dir=sd)
        self.assertEqual((code, calls("kernels", "push")), (kp.EXIT_PRIVACY, []), out[-400:])

    def test_push_error_is_registered(self):
        sd, code, out, entry = self.push("nb_push_error", push_error=True)
        self.assertEqual(code, kp.EXIT_USAGE, out[-400:])
        self.assertEqual(entry["status"], "push_failed")

    def test_kernel_output_prints_the_runs_own_outcome(self):
        sd = fresh_state_dir("nb_output")
        server(output_files={"step4_lora/train_summary.json": json.dumps({"status": "paused", "protocol_run": True, "chosen_epoch": 1,
                                                                           "epochs_completed": 2, "epochs_planned": 4}),
                             "dev__cte_p1__none__tier1__lora-12345678.status.json": json.dumps({"status": "incomplete", "n_done": 120, "n_jobs": 300, "message": "time budget"}),
                             "ok.status.json": json.dumps({"status": "complete", "n_done": 20, "n_jobs": 20})})
        dest = os.path.join(TMP, "kernel_out")
        code, out = run("kernel-output", "--slug", "train", "--dest", dest, state_dir=sd)
        self.assertEqual(code, 0, out[-600:])
        self.assertIn("NOT FINISHED  step4_lora/train_summary.json: status='paused'", out)
        self.assertIn("120/300 jobs", out)
        self.assertIn("OK    ok.status.json", out)
        self.assertIn("is NOT success by itself", out)
        code, out = run("kernel-status", "--slug", "train", state_dir=sd)
        self.assertIn("'complete' only means the script exited with status 0", out)


class Guard(unittest.TestCase):
    def test_nothing_public_can_be_run(self):
        server()
        R = kp.Runner(False, STUB)
        for args, code in ((("datasets", "create", "-p", "x", "--public"), kp.EXIT_PRIVACY), (("datasets", "create", "-p", "x", "-u"), kp.EXIT_PRIVACY),
                           (("datasets", "metadata", "o/s", "--update"), kp.EXIT_PRIVACY), (("datasets", "delete", "o/s"), kp.EXIT_USAGE),
                           (("kernels", "delete", "o/s"), kp.EXIT_USAGE), (("datasets", "download", "o/s"), kp.EXIT_USAGE),
                           (("models", "create"), kp.EXIT_USAGE)):
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
                R.kaggle(*args)
            self.assertEqual(cm.exception.code, code, args)
        self.assertEqual(state()["calls"], [])
        self.assertEqual(json.loads(kp.dataset_metadata("o", "mhist-priv-x", None, "Private working data, owner only", "d"))["isPrivate"], True)
        meta = json.loads(kp.kernel_metadata("o", "mhist-priv-x", "s.py", [], [], False, "NvidiaTeslaT4"))
        self.assertEqual((meta["is_private"], meta["enable_internet"]), ("true", "false"))
        for argv in (("kernel-push", "--script", os.path.join(HERE, "train_lora.py"), "--slug", "public-thing"),
                     ("kernel-push", "--script", os.path.join(HERE, "train_lora.py"), "--slug", "mhist-priv-x", "--dataset", "someone/else-data"),
                     ("kernel-push", "--script", os.path.join(HERE, "train_lora.py"), "--slug", "mhist-priv-x", "--accelerator", "NvidiaTeslaP100")):
            self.assertEqual(run(*argv, dry=True)[0], kp.EXIT_USAGE, argv)


class Secrets(unittest.TestCase):
    def push_text(self, text, *extra):
        path = os.path.join(TMP, "secret_script.py")
        with open(path, "w") as fh:
            fh.write(text)
        server()
        return run("kernel-push", "--script", path, "--slug", "mhist-priv-probe", *extra, state_dir=fresh_state_dir("secrets"))

    def test_key_formats_are_caught_and_never_echoed(self):
        self.assertEqual(kp.secret_hits("print('nothing here')  # sk- hf_ AIza"), [])
        for family, value in FAKES.items():
            hits = kp.secret_hits(f"X = '{value}'")
            self.assertEqual(len(hits), 1, f"{family}: {hits}")
            code, out = self.push_text(f'"""doc"""\nKEY = "{value}"\n')
            self.assertEqual((code, calls("kernels", "push")), (kp.EXIT_LEAK, []), family)
            # the family name carries the public prefix of the format (e.g. "sk-proj-..."); nothing after it is printed
            self.assertNotIn(value[-10:], out, f"{family}: part of the secret was printed")
            self.assertNotIn(value[len(value) // 2:len(value) // 2 + 8], out, f"{family}: part of the secret was printed")

    def test_this_machines_own_credentials(self):
        cases = {"kaggle.json": FAKE_KAGGLE_KEY, ".env": FAKE_ENV_VALUE}
        os.environ["KAGGLE_KEY"] = "1e" * 16
        os.environ["KAGGLE_API_TOKEN"] = "KG" + "x7" * 14
        with open(os.path.join(CFG, "access_token"), "w") as fh:
            fh.write(FAKE_ACCESS_TOKEN + "\n")
        cases.update({"KAGGLE_KEY": os.environ["KAGGLE_KEY"], "KAGGLE_API_TOKEN": os.environ["KAGGLE_API_TOKEN"], "access_token": FAKE_ACCESS_TOKEN})
        try:
            for what, value in cases.items():
                code, out = self.push_text(f'"""doc"""\nV = "{value}"\n')
                self.assertEqual((code, calls("kernels", "push")), (kp.EXIT_LEAK, []), what)
                self.assertIn(what if what != ".env" else "FRIENDLI_TOKEN", out)
                self.assertNotIn(value, out)
                self.assertNotIn(value[:6], out)
                # the same value passed as an argument or an environment variable for the script
                code, out = self.push_text('"""doc"""\n', f"--args=--tag {value}")
                self.assertEqual((code, calls("kernels", "push")), (kp.EXIT_LEAK, []), what + " via --args")
                code, out = self.push_text('"""doc"""\n', "--env", f"SETTING={value}")
                self.assertEqual((code, calls("kernels", "push")), (kp.EXIT_LEAK, []), what + " via --env")
                self.assertNotIn(value, out)
            # in a dataset upload too
            server()
            code, out = run("dataset-create", "--kind", "adapter", "--dir", adapter_dir("adapter_secret", {"README.md": f"token {FAKE_ENV_VALUE}\n"}),
                            state_dir=fresh_state_dir("secrets_ds"))
            self.assertEqual((code, calls("datasets", "create")), (kp.EXIT_LEAK, []), out[-300:])
            self.assertNotIn(FAKE_ENV_VALUE, out)
        finally:
            os.environ.pop("KAGGLE_KEY", None)
            os.environ.pop("KAGGLE_API_TOKEN", None)
            os.remove(os.path.join(CFG, "access_token"))
            kp._OWN_SECRETS = None
        self.assertEqual(self.push_text('"""doc"""\nSHORT = "abc"\n')[0], 0)               # a short .env value is not a secret

    def test_env_names_that_look_like_credentials(self):
        for name in ("HF_TOKEN", "OPENROUTER_API_KEY", "MY_SECRET", "DB_PASSWORD", "kaggle_key"):
            code, out = self.push_text('"""doc"""\n', "--env", f"{name}=harmless")
            self.assertEqual((code, calls("kernels", "push")), (kp.EXIT_USAGE, []), name)
            self.assertIn("looks like a credential", out)
        code, out = self.push_text('"""doc"""\n', "--env", "MHIST_FT_OUT_DIR=/kaggle/working/x")
        self.assertEqual(code, 0, out[-400:])
        self.assertEqual(registry(os.path.join(TMP, "state_secrets"))["kernels"][f"{OWNER}/mhist-priv-probe"]["env"], {"MHIST_FT_OUT_DIR": "/kaggle/working/x"})

    def test_credentials_file_is_checked_without_showing_the_key(self):
        code, out = run("check", state_dir=fresh_state_dir("check"))
        self.assertEqual(code, 0, out[-400:])
        self.assertIn(f"username: {OWNER}", out)
        self.assertIn("check passed", out)
        self.assertNotIn(FAKE_KAGGLE_KEY, out)
        os.chmod(os.path.join(CFG, "kaggle.json"), 0o644)
        try:
            code, out = run("check", state_dir=fresh_state_dir("check"))
            self.assertEqual(code, kp.EXIT_USAGE)
            self.assertIn("chmod 600", out)
        finally:
            os.chmod(os.path.join(CFG, "kaggle.json"), 0o600)


if __name__ == "__main__":
    unittest.main(verbosity=2 if "-v" in sys.argv else 1)
