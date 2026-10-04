#!/usr/bin/env python3
"""PRIVATE-only wrapper around the Kaggle CLI for step 4 (runs/competence/PLAN.md, "Addendum: step 4").

Every dataset and notebook this tool creates is private, and that is checked against Kaggle after each change.
There is no option that makes anything public. Nothing here runs unless you run it; add --dry-run to any
subcommand to print the exact kaggle commands and metadata files without touching the network or the disk.

    python3 kaggle_ft/kaggle_push.py check                      credentials file (600, username only) + CLI + quota
    python3 kaggle_ft/kaggle_push.py dataset-create             upload the data bundle as a new private dataset
    python3 kaggle_ft/kaggle_push.py dataset-create --kind jobs|adapter|wheels --dir DIR    upload another folder
    python3 kaggle_ft/kaggle_push.py dataset-version -m NOTES   upload a new version of an existing private dataset
    python3 kaggle_ft/kaggle_push.py weights                    upload the local MedGemma copy as a private dataset
    python3 kaggle_ft/kaggle_push.py kernel-push --script F --slug S [--dataset D ...] [--args="..."]
    python3 kaggle_ft/kaggle_push.py kernel-status --slug S [--wait]
    python3 kaggle_ft/kaggle_push.py kernel-output --slug S     download outputs to ~/mhist_local/kaggle_out/S/
    python3 kaggle_ft/kaggle_push.py verify-private             assert everything this tool created is still private
    python3 kaggle_ft/kaggle_push.py dataset-status --slug S | quota

Each subcommand has its own --help. Defaults: bundle ~/mhist_local/kaggle_bundle_mhist -> dataset
<you>/mhist-priv-bundle; weights ~/mhist_local/medgemma-1.5-4b-it -> dataset <you>/mhist-priv-medgemma15-4b.

How privacy is enforced
    * datasets: `kaggle datasets create` is private unless -u/--public is passed. This tool never passes it and
      refuses to run any command line that contains it (or `--update`, which can flip a dataset to public).
    * notebooks: kernel-metadata.json always carries "is_private": "true".
    * after every create / version / push the tool asks Kaggle for the resource's metadata and stops with exit
      code 3 and a loud message unless it says private. verify-private repeats that for everything in the local
      registry and for every dataset / notebook on the account whose slug starts with "mhist-priv-", and also
      looks at each page without credentials, the way a stranger would.
    * notebooks run with internet OFF. If Kaggle reports it ON although OFF was asked for, that is treated like a
      privacy failure (exit code 3). Internet ON needs both --enable-internet true and --i-accept-internet.
    * before every upload the directory (or script) is scanned: no test-partition tile, no label for a dev tile,
      no API token. Uploads are allow-listed by kind: the data bundle must match its MANIFEST.json exactly; a
      jobs folder may hold nothing but valid label-free job files; an adapter folder exactly the adapter files;
      a wheels folder only *.whl. Anything else, and any file whose content cannot be read, is refused.

Exit codes: 0 ok, 1 usage or precondition, 2 leak tripwire, 3 PRIVACY CHECK FAILED.
Standard library only. Wraps the Kaggle CLI at ~/mhist_local/venv/bin/kaggle (tested against kaggle 2.2.4).
"""

import argparse
import ast
import csv
import glob
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_bundle as bb  # noqa: E402  (standard library only at import; provides the leak tripwire)

HOME = os.path.expanduser("~")
LOCAL = os.environ.get("MHIST_LOCAL") or os.path.join(HOME, "mhist_local")
KAGGLE_BIN = os.environ.get("KAGGLE_BIN") or os.path.join(LOCAL, "venv", "bin", "kaggle")
WEIGHTS_DIR = os.path.join(LOCAL, "medgemma-1.5-4b-it")
STATE_DIR = os.environ.get("MHIST_KAGGLE_STATE") or os.path.join(LOCAL, "kaggle_stage")
OUT_DIR = os.path.join(LOCAL, "kaggle_out")

PREFIX = "mhist-priv-"
SLUGS = {"bundle": PREFIX + "bundle", "weights": PREFIX + "medgemma15-4b", "jobs": PREFIX + "jobs",
         "adapter": PREFIX + "adapter", "wheels": PREFIX + "wheels", "train": PREFIX + "train", "infer": PREFIX + "infer"}
# What a dataset upload may hold, by kind (fnmatch patterns over relative paths). The data bundle is held to its
# MANIFEST.json instead, and the weights to the file list of the pinned revision (cmd_weights).
UPLOAD_KINDS = ("bundle", "jobs", "adapter", "wheels")
KIND_ALLOW = {"jobs": ["*.jsonl"],
              "adapter": ["adapter_config.json", "adapter_model.safetensors", "step4_meta.json", "README.md"],
              "wheels": ["*.whl"]}
ADAPTER_REQUIRED = ("adapter_config.json", "adapter_model.safetensors", "step4_meta.json")
JOB_IMAGE = re.compile(r"^gridded/MHIST_[a-z]{3}\.png$")
MODEL_ID, MODEL_REVISION = "google/medgemma-1.5-4b-it", "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"
# `machine_shape` values (kaggle-cli docs/kernels.md, Sep 2026). NvidiaTeslaT4 is "GPU T4 x2", the default GPU.
ACCELERATORS = ("NvidiaTeslaT4", "NvidiaL4", "NvidiaL4X1", "none")
RETIRED_ACCELERATORS = {"NvidiaTeslaP100": "retired 2026-09-14: Kaggle silently runs it on NvidiaTeslaT4, and current "
                                           "torch builds have no kernels for the P100 (sm_60)"}
KAGGLE_INPUT = "/kaggle/input"

EXIT_USAGE, EXIT_LEAK, EXIT_PRIVACY = 1, 2, 3
USER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,49}$")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{4,48}[a-z0-9]$")
# Credential formats, by family. A hit is reported by its family name only: no part of the match is ever printed.
SECRET_FAMILIES = [(name, re.compile(pat)) for name, pat in (
    ("a Hugging Face token (hf_...)", r"hf_[A-Za-z0-9]{30,}"),
    ("a Kaggle API token (KGAT_...)", r"KGAT_[A-Za-z0-9_]{20,}"),
    ("an OpenRouter key (sk-or-v1-...)", r"sk-or-v1-[A-Za-z0-9]{10,}"),
    ("an Anthropic key (sk-ant-...)", r"sk-ant-[A-Za-z0-9_-]{20,}"),
    ("an OpenAI project key (sk-proj-...)", r"sk-proj-[A-Za-z0-9_-]{20,}"),
    ("an OpenAI-style key (sk-...)", r"sk-[A-Za-z0-9]{30,}"),
    ("a Cerebras key (csk-...)", r"csk-[a-z0-9]{20,}"),
    ("a GitHub token (gh?_...)", r"gh[opsu]_[A-Za-z0-9]{20,}"),
    ("an AWS access key id (AKIA...)", r"AKIA[0-9A-Z]{16}"),
    ("a Google API key (AIza...)", r"AIza[0-9A-Za-z_-]{35}"),
    ("a Friendli token (flp_...)", r"flp_[A-Za-z0-9]{20,}"),
    ("a Slack token (xox?-...)", r"xox[abprs]-[A-Za-z0-9-]{10,}"),
    ("a private key block", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"))]
SECRET_NAME = re.compile(r"(?i)token|key|secret|password|passwd|credential")
# The project's own key file. Its values are compared with what is about to be uploaded and never shown.
# MHIST_DOTENV names another file; an empty value switches the comparison off.
DOTENV = os.environ["MHIST_DOTENV"] if "MHIST_DOTENV" in os.environ else os.path.join(bb.ROOT, ".env")
# The only Kaggle CLI commands this tool can run. No delete, no metadata --update, nothing public.
ALLOWED = {("datasets", "create"), ("datasets", "version"), ("datasets", "status"), ("datasets", "metadata"),
           ("datasets", "list"), ("kernels", "push"), ("kernels", "status"), ("kernels", "output"),
           ("kernels", "pull"), ("kernels", "list"), ("config", "view"), ("quota",)}
FORBIDDEN_FLAGS = {"-u", "--public", "--update"}
IGNORE = [".DS_Store", "._*"]


def say(msg=""):
    print(msg, flush=True)


def die(msg, code=EXIT_USAGE):
    print("STOP: " + msg, file=sys.stderr, flush=True)
    sys.exit(code)


def privacy_alarm(lines):
    bar = "!" * 100
    print("\n".join(["", bar, "PRIVACY CHECK FAILED - ACT NOW", *lines, bar]), file=sys.stderr, flush=True)
    sys.exit(EXIT_PRIVACY)


def sha256_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


# --------------------------------------------------------------------------------------------- credentials

def config_dir():
    return os.environ.get("KAGGLE_CONFIG_DIR") or os.path.join(HOME, ".kaggle")


def _account_cache():
    return os.path.join(HOME, "mhist_local", "kaggle_stage", "account.json")


def cached_username():
    try:
        with open(_account_cache()) as fh:
            u = json.load(fh).get("username")
        return u if isinstance(u, str) and USER_RE.match(u) else None
    except (OSError, ValueError):
        return None


def read_credentials():
    """Look at the credentials file without ever returning, printing or logging the key."""
    path = os.path.join(config_dir(), "kaggle.json")
    c = {"path": path, "exists": os.path.isfile(path), "mode": None, "mode_ok": False, "username": None,
         "has_key": False, "problems": [], "notes": []}
    tok_path = os.path.join(config_dir(), "access_token")
    if not c["exists"] and os.path.isfile(tok_path):
        # New-style Kaggle API token (settings page -> "Create New Token" gives a KGAT_ string, not kaggle.json).
        # The CLI reads it from ~/.kaggle/access_token; the username is learned from the CLI by `check` and cached.
        c.update(path=tok_path, exists=True, kind="access_token")
        c["mode"] = stat.S_IMODE(os.stat(tok_path).st_mode)
        c["mode_ok"] = c["mode"] & 0o077 == 0
        if not c["mode_ok"]:
            c["problems"].append(f"{tok_path} has permissions {c['mode']:03o}, readable by others; run: chmod 600 {shlex.quote(tok_path)}")
        try:
            with open(tok_path) as fh:
                c["has_key"] = len(fh.read().strip()) >= 16
        except OSError:
            c["has_key"] = False
        if not c["has_key"]:
            c["problems"].append(f"{tok_path} is empty or too short")
        c["username"] = cached_username()
        if not c["username"]:
            c["notes"].append("username not cached yet: run `kaggle_push.py check` once (it asks the CLI which account the token belongs to)")
    elif not c["exists"]:
        c["problems"].append(f"{path} does not exist")
    else:
        c["mode"] = stat.S_IMODE(os.stat(path).st_mode)
        c["mode_ok"] = c["mode"] & 0o077 == 0
        if not c["mode_ok"]:
            c["problems"].append(f"{path} has permissions {c['mode']:03o}, readable by others; run: chmod 600 {shlex.quote(path)}")
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            data = None
            c["problems"].append(f"{path} is not valid JSON (expected {{\"username\": ..., \"key\": ...}})")
        if isinstance(data, dict):
            u, k = data.get("username"), data.get("key")
            c["username"] = u if isinstance(u, str) and USER_RE.match(u) else None
            c["has_key"] = isinstance(k, str) and len(k) >= 16
            if not c["username"]:
                c["problems"].append(f'{path} has no valid "username" field')
            if not c["has_key"]:
                c["problems"].append(f'{path} has no "key" field')
        del data
    if c.get("kind") != "access_token" and os.path.isfile(os.path.join(config_dir(), "access_token")):
        c["notes"].append("~/.kaggle/access_token also exists: the CLI prefers it over kaggle.json, so `check` "
                          "(without --dry-run) confirms which account it logs in as")
    for v in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        if os.environ.get(v):
            c["notes"].append(f"environment variable {v} is set and overrides the credentials file (value not shown)")
    return c


_OWN_SECRETS = None


def own_secrets():
    """[(what it is, value)] for every credential this machine holds that could end up in an upload: the key in
    kaggle.json, KAGGLE_KEY / KAGGLE_API_TOKEN, ~/.kaggle/access_token and every value of 16 or more characters
    in the project's .env. They are read for comparison only: never printed, logged or written anywhere."""
    global _OWN_SECRETS
    if _OWN_SECRETS is not None:
        return _OWN_SECRETS
    found = []
    try:
        with open(os.path.join(config_dir(), "kaggle.json")) as fh:
            found.append(("the Kaggle API key from kaggle.json", json.load(fh).get("key")))
    except (OSError, ValueError, AttributeError):
        pass
    for v in ("KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        found.append((f"the value of the environment variable {v}", os.environ.get(v)))
    try:
        with open(os.path.join(config_dir(), "access_token")) as fh:
            found.append(("the Kaggle access token (access_token file)", fh.read().strip()))
    except OSError:
        pass
    if DOTENV:
        try:
            with open(DOTENV, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    name, value = line.split("=", 1)
                    name = name.replace("export ", "").strip()
                    value = value.strip()
                    for v in {value.strip("'\""), (value.split() or [""])[0].strip("'\"")}:
                        found.append((f"the value of {name} in the project's .env", v))
        except OSError:
            pass
    _OWN_SECRETS = [(what, v) for what, v in found if isinstance(v, str) and len(v) >= 16]
    return _OWN_SECRETS


def kaggle_version_offline(bin_path):
    venv = os.path.dirname(os.path.dirname(os.path.abspath(bin_path)))
    hits = sorted(glob.glob(os.path.join(venv, "lib", "python*", "site-packages", "kaggle-*.dist-info")))
    return os.path.basename(hits[-1])[len("kaggle-"):-len(".dist-info")] if hits else None


# ------------------------------------------------------------------------------------------------- runner

class Runner:
    """Runs (or, with dry=True, only prints) Kaggle CLI commands. Refuses anything outside the allow-list."""

    def __init__(self, dry, kaggle_bin):
        self.dry, self.bin = dry, kaggle_bin

    def _guard(self, args):
        bad = FORBIDDEN_FLAGS & set(args)
        if bad:
            die(f"refusing to run a Kaggle command with {sorted(bad)}: this tool only creates PRIVATE resources", EXIT_PRIVACY)
        if tuple(args[:2]) not in ALLOWED and tuple(args[:1]) not in ALLOWED:
            die(f"refusing to run `kaggle {' '.join(args[:2])}`: not on this tool's allow-list", EXIT_USAGE)

    def kaggle(self, *args, note="", tee=False, quiet=False):
        """-> (returncode, output). In a dry run nothing is executed and (None, '') is returned.
        quiet=True leaves out the command echo (repeated polls)."""
        self._guard(args)
        line = shlex.join([self.bin, *args])
        if self.dry:
            say(f"  $ {line}" + (f"\n        # {note}" if note else ""))
            return None, ""
        if not (os.path.isfile(self.bin) and os.access(self.bin, os.X_OK)):
            die(f"Kaggle CLI not found at {self.bin} (set KAGGLE_BIN or pass --kaggle-bin)")
        if not quiet:
            say(f"+ {line}")
        if not tee:
            p = subprocess.run([self.bin, *args], capture_output=True, text=True)
            return p.returncode, (p.stdout or "") + (p.stderr or "")
        # long uploads: progress (stderr) goes straight to the terminal, stdout is shown and kept
        p = subprocess.Popen([self.bin, *args], stdout=subprocess.PIPE, text=True, bufsize=1)
        out = []
        for ln in p.stdout:
            sys.stdout.write(ln)
            sys.stdout.flush()
            out.append(ln)
        return p.wait(), "".join(out)

    def write(self, path, text, what):
        if self.dry:
            say(f"  would write {what}: {path}\n" + "".join("      " + ln + "\n" for ln in text.splitlines()).rstrip("\n"))
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
        say(f"wrote {what}: {path}")


def header(a, R, title):
    v = kaggle_version_offline(R.bin)
    if a.dry_run:
        say(f"[dry-run] {title}: nothing is sent to Kaggle and no file is written")
    say(f"kaggle CLI: {R.bin} " + (f"(version {v})" if v else "(version unknown: not found next to a site-packages dist-info)"))
    if v and tuple(int(x) for x in re.findall(r"\d+", v)[:2]) < (2, 0):
        die(f"kaggle {v} is too old: this tool needs kaggle >= 2.0 (--dir-mode, --ignore-patterns, kernels pull -m)")


def owner_of(a):
    """The Kaggle username that owns everything. From --owner, else KAGGLE_USERNAME, else kaggle.json."""
    c = read_credentials()
    owner = a.owner or os.environ.get("KAGGLE_USERNAME") or c["username"]
    if a.owner and c["username"] and a.owner != c["username"]:
        die(f"--owner {a.owner} is not the account in {c['path']} ({c['username']}): this tool only writes to your own account")
    if not a.dry_run and c["problems"] and not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        die("credentials are not ready:\n  - " + "\n  - ".join(c["problems"]) + "\nRun `kaggle_push.py check` for instructions.")
    if not owner:
        if not a.dry_run:
            die("no Kaggle username: run `kaggle_push.py check`")
        owner = "YOUR_KAGGLE_USERNAME"
        say(f"owner: {owner}  (placeholder: {c['path']} does not exist yet; the real run reads the username from it)")
    else:
        if not USER_RE.match(owner):
            die(f"not a valid Kaggle username: {owner!r}")
        say(f"owner: {owner}")
    return owner


def check_slug(slug, a):
    if not SLUG_RE.match(slug):
        die(f"bad slug {slug!r}: 6-50 characters, lowercase letters, digits and dashes")
    if not slug.startswith(PREFIX) and not getattr(a, "any_slug", False):
        die(f"slug {slug!r} must start with {PREFIX!r} so that verify-private can find it (override: --any-slug)")
    return slug


def to_ref(owner, name, a):
    """'bundle' / 'weights' / a slug / 'owner/slug' -> 'owner/slug'. Other owners are refused unless --allow-foreign."""
    name = SLUGS.get(name, name)
    ref = name if "/" in name else f"{owner}/{name}"
    o, s = ref.split("/", 1)
    if o != owner and not getattr(a, "allow_foreign", False):
        die(f"{ref} belongs to another account; this tool attaches only your own private data (override: --allow-foreign)")
    if not SLUG_RE.match(s):
        die(f"bad slug in {ref!r}")
    return ref


# ------------------------------------------------------------------------------------------------ registry

def registry_path(a):
    return os.path.join(a.state_dir, "registry.json")


def load_registry(a):
    try:
        with open(registry_path(a)) as fh:
            r = json.load(fh)
    except (OSError, ValueError):
        r = {}
    r.setdefault("datasets", {})
    r.setdefault("kernels", {})
    return r


def save_registry(a, reg):
    os.makedirs(a.state_dir, exist_ok=True)
    tmp = registry_path(a) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(reg, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, registry_path(a))


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# ---------------------------------------------------------------------------------------- privacy checks

def scratch(a):
    d = os.path.join(a.state_dir, "tmp")
    os.makedirs(d, exist_ok=True)
    return tempfile.mkdtemp(prefix="verify-", dir=d)


def dataset_privacy(a, R, ref):
    """Ask Kaggle for the dataset's metadata. -> 'private' | 'PUBLIC' | 'SHARED' | 'missing' | 'unknown' | 'dry'.
    SHARED = private but with collaborators, which breaks "my account only"."""
    if R.dry:
        R.kaggle("datasets", "metadata", ref, "-p", os.path.join(a.state_dir, "tmp", "verify-XXXX"),
                 note='the downloaded dataset-metadata.json must contain "isPrivate": true (a public dataset has no such key) and no collaborators')
        return "dry", ""
    d = scratch(a)
    try:
        rc, out = R.kaggle("datasets", "metadata", ref, "-p", d)
        path = os.path.join(d, "dataset-metadata.json")
        if rc != 0 or not os.path.isfile(path):
            return ("missing" if re.search(r"\b404\b|not found", out, re.I) else "unknown"), out.strip()[-300:]
        with open(path) as fh:
            data = json.load(fh)
        info = data.get("info") if isinstance(data.get("info"), dict) else data
        owner = info.get("ownerUser")
        if owner and owner != ref.split("/")[0]:
            return "unknown", f"metadata is for owner {owner!r}"
        if info.get("isPrivate") is not True:
            return "PUBLIC", f"isPrivate={info.get('isPrivate')!r}"
        if info.get("collaborators"):
            return "SHARED", f"{len(info['collaborators'])} collaborator(s)"
        return "private", "isPrivate=True, no collaborators"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def kernel_privacy(a, R, ref):
    """Pull the notebook's server-side metadata. -> (state, metadata dict)."""
    if R.dry:
        R.kaggle("kernels", "pull", ref, "-p", os.path.join(a.state_dir, "tmp", "verify-XXXX"), "-m",
                 note='the downloaded kernel-metadata.json must contain "is_private": true')
        return "dry", {}
    d = scratch(a)
    try:
        rc, out = R.kaggle("kernels", "pull", ref, "-p", d, "-m")
        path = os.path.join(d, "kernel-metadata.json")
        if rc != 0 or not os.path.isfile(path):
            return ("missing" if re.search(r"\b404\b|not found", out, re.I) else "unknown"), {"detail": out.strip()[-300:]}
        with open(path) as fh:
            meta = json.load(fh)
        return ("private" if meta.get("is_private") is True else "PUBLIC"), meta
    finally:
        shutil.rmtree(d, ignore_errors=True)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def anon_status(url):
    """HTTP status of a GET with no credentials, no cookies and no redirects (None on a network error).
    curl is used when present: python.org builds of Python on macOS often have no CA certificates installed."""
    agent = "Mozilla/5.0 (privacy self-check, anonymous)"
    curl = shutil.which("curl")
    if curl:
        p = subprocess.run([curl, "-q", "-sS", "-m", "25", "-o", os.devnull, "-w", "%{http_code}", "-A", agent, url],
                           capture_output=True, text=True)
        code = p.stdout.strip()
        return int(code) if p.returncode == 0 and code.isdigit() and code != "000" else None
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, headers={"User-Agent": agent})
    try:
        with opener.open(req, timeout=25) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return None


def page_url(kind, ref):
    return f"https://www.kaggle.com/{'datasets' if kind == 'dataset' else 'code'}/{ref}"


def anon_probe(kind, ref, control_status):
    """A stranger's view. 'hidden' = not served; 'VISIBLE' = served although a made-up slug is not; else 'inconclusive'."""
    s = anon_status(page_url(kind, ref))
    if s is None:
        return "inconclusive", "network error"
    if s != 200:
        return "hidden", f"HTTP {s}"
    if control_status == 200:
        return "inconclusive", "HTTP 200, but Kaggle also answers 200 for a slug that does not exist"
    return "VISIBLE", f"HTTP 200 (a made-up slug gives {control_status})"


def alarm_dataset(ref, detail):
    privacy_alarm([f"Kaggle does not report the dataset {ref} as private to your account only ({detail}).",
                   f"  1. Open https://www.kaggle.com/datasets/{ref}/settings: set Visibility to Private and remove every collaborator, or delete it:",
                   f"       {KAGGLE_BIN} datasets delete {ref}",
                   "  2. Then run: python3 kaggle_ft/kaggle_push.py verify-private",
                   "Nothing further was done."])


def alarm_kernel(ref, detail):
    privacy_alarm([f"Kaggle does not report the notebook {ref} as private ({detail}).",
                   f"  1. Open https://www.kaggle.com/code/{ref} -> Share -> Private, or delete it:",
                   f"       {KAGGLE_BIN} kernels delete {ref}",
                   "  2. Then run: python3 kaggle_ft/kaggle_push.py verify-private",
                   "Nothing further was done."])


# --------------------------------------------------------------------------------------- local tripwires

def secret_hits(text):
    """What kind of secret the text contains, as family names only (never a character of the match)."""
    hits = [name for name, pat in SECRET_FAMILIES if pat.search(text)]
    for what, value in own_secrets():
        if value in text and what not in hits:
            hits.append(what)
    return hits


def kind_problems(folder, kind, files):
    """Structure rules of one upload kind, beyond the file-name allow-list. Returns a list of problems."""
    out = []
    if kind in KIND_ALLOW:
        nested = [f for f in files if "/" in f]
        if nested:
            out.append(f"{nested[0]}: a {kind} upload is one flat folder; {len(nested)} files lie in sub-directories")
    if kind == "jobs":
        import infer_jobs as ij                              # standard library only at import
        n_jobs = 0
        for rel in files:
            if not rel.endswith(".jsonl") or "/" in rel:
                continue
            with open(os.path.join(folder, rel), encoding="utf-8", errors="replace") as fh:
                for n, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    try:
                        job = ij.validate_job(json.loads(line), f"{rel} line {n}")
                        bad = [p_["file"] for p_ in job["parts"] if p_["type"] == "image" and not JOB_IMAGE.match(p_["file"])]
                        if bad:
                            raise ValueError(f"{rel} line {n}: image file {bad[0]!r} is not gridded/MHIST_xxx.png")
                    except ValueError as e:                  # includes json.JSONDecodeError
                        out.append(f"{rel}: not a job file ({str(e)[:240]})")
                        break
                    n_jobs += 1
        if not n_jobs and not out:
            out.append("no job file (*.jsonl written by build_dev_jobs.py) in this folder")
    elif kind == "adapter":
        for name in ADAPTER_REQUIRED:
            if name not in files:
                out.append(f"{name} is missing: an adapter upload is train_lora.py's best_adapter/ folder, unchanged")
        for name in ("adapter_config.json", "step4_meta.json"):
            if name in files:
                try:
                    with open(os.path.join(folder, name)) as fh:
                        if not isinstance(json.load(fh), dict):
                            raise ValueError("not a JSON object")
                except (OSError, ValueError) as e:
                    out.append(f"{name}: cannot be read as a JSON object ({type(e).__name__})")
    elif kind == "wheels":
        if not files:
            out.append("no *.whl file in this folder")
    return out


def audit_upload_dir(folder, kind, only=None, allow=None):
    """Leak tripwire + secret scan on a directory about to be uploaded. Local, read-only. Exits 2 on a hit.

    kind: 'bundle' (held to its MANIFEST.json), 'jobs' / 'adapter' / 'wheels' (KIND_ALLOW plus kind_problems), or
    'weights' (the caller passes the exact file list as `allow`). There is no kind that accepts arbitrary files.
    `only` limits the audit to these relative paths (the files a staging step would pick out of a larger folder)."""
    truth = bb.load_truth()
    files = [f for f in bb.bundle_files(folder) if only is None or f in only]
    if kind == "bundle":
        problems, man = bb.verify_bundle(folder, truth)
        fingerprint = bb.sha256_file(os.path.join(folder, "MANIFEST.json")) if man else None
    else:
        allow = allow if allow is not None else KIND_ALLOW.get(kind)
        if allow is None:
            die(f"internal: no allow-list for upload kind {kind!r}")
        problems = bb.audit_dir(folder, truth, allow=allow, only=None if only is None else set(only))
        problems += kind_problems(folder, kind, files)
        h = hashlib.sha256()
        for rel in files:
            full = os.path.join(folder, rel)
            h.update(f"{rel}\0{os.path.getsize(full)}\0".encode())
            if os.path.getsize(full) <= 256 << 20:              # big weight files: name + size only
                h.update(bb.sha256_file(full).encode())
        fingerprint = h.hexdigest()
    for rel in files:
        full = os.path.join(folder, rel)
        if bb._is_text(full, rel, 64 << 20):
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                for hit in secret_hits(fh.read()):
                    problems.append(f"{rel}: looks like it contains a secret ({hit})")
    if problems:
        print(f"LEAK TRIPWIRE: {folder} is NOT uploaded (checked as: {kind}).", file=sys.stderr)
        for p in problems[:40]:
            print("  - " + p, file=sys.stderr)
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more", file=sys.stderr)
        sys.exit(EXIT_LEAK)
    return fingerprint


# ------------------------------------------------------------------------------------------------ datasets

def dataset_metadata(owner, slug, title, subtitle, description):
    """dataset-metadata.json. Privacy is NOT a key the create command reads (it is the absence of --public);
    "isPrivate": true is written anyway so that a stray `kaggle datasets metadata --update` cannot flip it."""
    md = {"title": title or slug, "id": f"{owner}/{slug}", "subtitle": subtitle, "description": description,
          "isPrivate": True, "licenses": [{"name": "other"}]}
    assert 6 <= len(md["title"]) <= 50 and 20 <= len(subtitle) <= 80 and md["isPrivate"] is True
    return json.dumps(md, indent=2) + "\n"


TERMINAL_BAD_STATUS = ("failed", "deleted")      # what kaggle 2.2.4 prints (DatabundleVersionStatus, lower-cased)


def wait_ready(a, R, ref, on_visible=None):
    """Poll until Kaggle has processed the dataset. -> 'ready' | 'error' | 'timeout' | 'dry'. It never exits:
    the caller checks privacy first and only then acts on a processing error.
    While Kaggle is still creating a dataset it answers 403 even to the owner. on_visible() is called once, the
    first time Kaggle answers the status request at all while the dataset is not ready yet: from then on the
    owner can read its metadata, so privacy is checked at that moment and not only at the end of the wait."""
    if R.dry:
        R.kaggle("datasets", "status", ref, note=f"polled every 15 s (up to {a.wait_min} min) until it prints: ready")
        return "dry"
    t0, last, polls = time.time(), None, 0
    while True:
        rc, out = R.kaggle("datasets", "status", ref, quiet=polls > 0)
        polls += 1
        prev, last = last, (out.strip().splitlines()[-1] if out.strip() else "")
        if last != prev:                                                 # a 403 here means: not created yet
            say(f"  status ({int(time.time() - t0)} s): {last[:160]}")
        word = last.lower()
        if rc == 0 and word == "ready":
            return "ready"
        if rc == 0 and (word in TERMINAL_BAD_STATUS or "error" in word):
            return "error"
        if rc == 0 and on_visible is not None:
            on_visible()
            on_visible = None
        if time.time() - t0 >= a.wait_min * 60:
            break
        time.sleep(getattr(a, "poll_seconds", 15))
    say(f"  still not ready after {a.wait_min} min (large uploads take a while); check later with: dataset-status --slug {ref.split('/')[1]}")
    return "timeout"


def anon_pending_evidence(ref):
    """Positive evidence only that a stranger cannot see `ref`: its page without credentials answers 404 AND a
    made-up slug of the same owner answers 404 too. -> (True | False, text). Any other answer (network error,
    200, 3xx, 403, 429, 5xx) proves nothing and is False."""
    control = anon_status(page_url("dataset", f"{ref.split('/')[0]}/{PREFIX}no-such-thing-{int(time.time()) % 100000}"))
    page = anon_status(page_url("dataset", ref))
    return (page == 404 and control == 404), f"HTTP {page} (a made-up slug gives {control})"


def confirm_dataset_private(a, R, ref, tries=8, pending_ok=False):
    """-> True when Kaggle reports the dataset private to the owner only. Anything else is a privacy alarm
    (exit 3), with one exception: pending_ok=True accepts "Kaggle cannot show it to its owner yet" and
    returns False, PROVIDED there is positive evidence that a stranger cannot see it either (the page without
    credentials answers 404, exactly like a made-up slug). Kaggle answers 403 to the owner's own token for
    a dataset it is still creating (seen 2026-10-04: 15 minutes for a 228 MB upload, during which the owner's
    list did not show it and the anonymous page was 404). PUBLIC / SHARED always alarm."""
    state, detail = "unknown", ""
    for i in range(tries):
        state, detail = dataset_privacy(a, R, ref)
        if state in ("private", "PUBLIC", "SHARED", "dry"):
            break
        time.sleep(getattr(a, "retry_seconds", 10))
    if state == "dry":
        return None
    if state == "private":
        say(f"PRIVATE confirmed by Kaggle: {ref} ({detail})")
        return True
    if pending_ok and state in ("unknown", "missing"):
        hidden, seen = anon_pending_evidence(ref)
        if hidden:
            say(f"Kaggle has not finished creating {ref} (its answer to the owner: {state}). The page without credentials "
                f"answers 404, the same as a made-up slug. Privacy is checked again as soon as Kaggle shows it to its owner.")
            return False
        detail = f"{detail}; anonymous view: {seen}"
    alarm_dataset(ref, f"{state}: {detail}")


def is_step4_bundle(folder):
    """True if the directory is (or claims to be) the step-4 data bundle; it is then held to its MANIFEST.json."""
    try:
        with open(os.path.join(folder, "MANIFEST.json")) as fh:
            if json.load(fh).get("bundle") == "mhist-step4-private":
                return True
    except (OSError, ValueError, AttributeError):
        pass
    if os.path.abspath(folder) == os.path.abspath(bb.DEFAULT_OUT) or os.path.isdir(os.path.join(folder, "gridded")):
        die(f"{folder} looks like the data bundle but has no valid MANIFEST.json: run build_bundle.py first")
    return False


def mirror(src, dst):
    """Make dst an exact copy of src (our own staging area: files that are no longer in src are removed)."""
    os.makedirs(dst, exist_ok=True)
    want = set(bb.bundle_files(src))
    for rel in bb.bundle_files(dst):
        if rel not in want:
            os.remove(os.path.join(dst, rel))
    for rel in sorted(want):
        s_, d_ = os.path.join(src, rel), os.path.join(dst, rel)
        os.makedirs(os.path.dirname(d_), exist_ok=True)
        if not (os.path.isfile(d_) and os.path.getsize(d_) == os.path.getsize(s_) and bb.sha256_file(d_) == bb.sha256_file(s_)):
            shutil.copy2(s_, d_)


def resolve_kind(a, src):
    """What the directory is uploaded as. The data bundle is recognised by its MANIFEST.json; for anything else
    --kind must say what it is, so that the matching allow-list is applied. There is no 'any folder' kind."""
    looks_bundle = is_step4_bundle(src)
    kind = a.kind or ("bundle" if looks_bundle else None)
    if kind is None:
        die(f"--kind is required for {src}: one of {', '.join(UPLOAD_KINDS)}. Each kind may hold only its own files "
            "(jobs: label-free *.jsonl job files; adapter: best_adapter/ unchanged; wheels: *.whl).")
    if (kind == "bundle") != looks_bundle:
        die(f"--kind {kind} does not fit {src}: " + ("it is the step-4 data bundle (it has that MANIFEST.json)" if looks_bundle
                                                     else "it has no step-4 MANIFEST.json; build it with build_bundle.py"))
    return kind


def upload_location(a, R, src, slug, kind):
    """Where the upload is made from. The data bundle and anything else outside iCloud-synced folders is uploaded
    in place. A directory inside Documents / Desktop / iCloud Drive is copied to the staging area first, so that
    the Kaggle metadata file (it names the account) is never written into the project folder. The caller has
    already audited `src`: nothing is copied anywhere before the leak tripwire has passed."""
    if kind == "bundle" or not bb.in_synced_folder(src):
        return src
    stage = os.path.join(a.state_dir, "datasets", slug)
    if R.dry:
        say(f"  (would first copy {src} to the staging area {stage} and upload from there)")
    else:
        mirror(src, stage)
        say(f"staged a copy of {src} in {stage}")
    return stage


def push_dataset(a, R, folder, slug, title, subtitle, description, dir_mode, kind, version_notes=None, extra_ignore=(),
                 mount_note="", owner=None, scan=None, allow=None):
    """`folder` is what the Kaggle CLI uploads; `scan` (default: the same) is what the local checks read. They
    differ only in a dry run of a staged directory, where the staging copy does not exist yet."""
    owner = owner or owner_of(a)
    ref = f"{owner}/{check_slug(slug, a)}"
    scan = scan or folder
    if not os.path.isdir(scan):
        die(f"{scan} is not a directory" + (" (run build_bundle.py first)" if kind == "bundle" else ""))
    is_bundle = kind == "bundle"
    fingerprint = audit_upload_dir(scan, kind, allow=allow)
    files = bb.bundle_files(scan)

    def register(**fields):
        """Record the dataset in the registry BEFORE Kaggle is asked to create it, so that verify-private looks
        at it whatever happens to this process afterwards."""
        if R.dry:
            return
        reg_ = load_registry(a)
        e_ = reg_["datasets"].setdefault(ref, {"created": now()})
        e_.update(fields)
        save_registry(a, reg_)
    total = sum(os.path.getsize(os.path.join(scan, f)) for f in files)
    top = sorted(os.listdir(scan))
    say(f"local checks passed: {scan}\n  {len(files)} files, {total / 1e6:.1f} MB; no test-partition tile, no dev label, no secret"
        + ("; matches MANIFEST.json" if is_bundle else "") + f"\n  content fingerprint {fingerprint}")
    n_top = sum(1 for f in top if f not in bb.NOT_BUNDLE)
    if n_top > 50:
        die(f"{n_top} top-level entries: Kaggle allows 50 top-level files per dataset; put files in sub-directories")
    if total > 190e9:
        die("over 190 GB: Kaggle's limit is 200 GB per dataset and 200 GB of private data per account")
    reg = load_registry(a)
    known = reg["datasets"].get(ref, {})
    ignore = [x for p in (*IGNORE, *extra_ignore) for x in ("--ignore-patterns", p)]
    meta_path = os.path.join(folder, "dataset-metadata.json")
    meta = dataset_metadata(owner, slug, title, subtitle, description)

    say("" if not R.dry else "\nplan:")
    if version_notes is None:                                           # ---- create
        rc, out = R.kaggle("datasets", "status", ref, note="must FAIL (403 / 404): the dataset must not exist yet")
        if rc == 0:
            if known.get("fingerprint") == fingerprint:
                say(f"{ref} already exists and the registry says it holds this exact content: not uploading again.")
                if confirm_dataset_private(a, R, ref):
                    word = out.strip().splitlines()[-1].lower() if out.strip() else ""
                    register(private_confirmed=now(), **({"status": "ready"} if word == "ready" else {}))
                return ref
            die(f"{ref} already exists on Kaggle. To upload new content use: dataset-version --slug {slug} -m '...'")
        resume = (not R.dry and known.get("fingerprint") == fingerprint and known.get("status") in ("uploaded", "processing"))
        if resume:
            # Kaggle answers 403 both for "does not exist" and for "still being created". The registry says this exact
            # content was uploaded completely: do not create it a second time, go back to waiting for it.
            say(f"{ref}: the registry says this exact content was uploaded at {known.get('last_upload')} and Kaggle has not shown "
                "it yet. Not uploading again; waiting for Kaggle to finish creating it. (If you deleted it on kaggle.com, "
                f"remove its entry from {registry_path(a)} first.)")
            rc, out, ok = None, "", ""
        else:
            R.write(meta_path, meta, "dataset-metadata.json")
            if R.dry:
                say(f"  would record {ref} in {registry_path(a)} with status 'uploading'  (before the upload, so that "
                    "verify-private knows it even if this process dies)")
            register(kind=kind, folder=folder, status="uploading", upload_started=now(), pending_fingerprint=fingerprint,
                     private_confirmed=None)
            rc, out = R.kaggle("datasets", "create", "-p", folder, "-r", dir_mode, *ignore, tee=True,
                               note="private: no -u/--public (this tool cannot pass it). " +
                                    ("Sub-directories are uploaded as zip archives and unpacked by Kaggle. " if dir_mode == "zip"
                                     else "Sub-directories are not uploaded. ") + "Re-running resumes an interrupted upload.")
            ok = "Your private Dataset is being created"
    else:                                                               # ---- new version
        same = not a.force and known.get("fingerprint") == fingerprint
        if same and (R.dry or known.get("status") not in ("uploaded", "processing")):
            say(f"{ref}: the registry says this exact content is already uploaded: nothing to do (override: --force).")
            if confirm_dataset_private(a, R, ref):
                register(private_confirmed=now())
            return ref
        if same:                                                        # uploaded, but Kaggle had not finished: wait again
            say(f"{ref}: the registry says this exact content was uploaded at {known.get('last_upload')} and Kaggle had not "
                "finished processing it. Not uploading again; waiting for it.")
            rc, out, ok = None, "", ""
        else:
            state, detail = dataset_privacy(a, R, ref)
            if state in ("PUBLIC", "SHARED"):
                alarm_dataset(ref, detail)
            if state in ("missing", "unknown"):
                die(f"cannot read {ref} ({state}: {detail}). If it does not exist yet, use dataset-create.")
            R.write(meta_path, meta, "dataset-metadata.json")
            register(kind=kind, folder=folder, status="uploading", upload_started=now(), pending_fingerprint=fingerprint,
                     private_confirmed=None)
            rc, out = R.kaggle("datasets", "version", "-p", folder, "-m", version_notes, "-r", dir_mode, *ignore,
                               *(["-d"] if a.delete_old_versions else []), tee=True,
                               note="a new version keeps the dataset's visibility; it was just confirmed private")
            ok = "Dataset version is being created"
    if rc is not None:
        if re.search(r"public dataset", out, re.I):
            register(status="ALARM: the Kaggle CLI printed 'public Dataset'")
            alarm_dataset(ref, "the Kaggle CLI printed 'public Dataset'")
        if rc != 0 or ok not in out:
            # The client failed, but the server may have created the dataset: look at it before leaving.
            register(status="upload_failed", upload_failed=now())
            state, detail = dataset_privacy(a, R, ref)
            if state in ("PUBLIC", "SHARED"):
                alarm_dataset(ref, detail)
            die(f"the upload did not succeed (exit code {rc}). Kaggle's answer about {ref} right now: {state}. It stays in "
                f"the registry, so verify-private keeps checking it. Re-run the same command to resume.")
        register(kind=kind, folder=folder, status="uploaded", fingerprint=fingerprint, files=len(files), bytes=total,
                 last_upload=now(), private_confirmed=None, version_notes=version_notes)
        say(f"registered in {registry_path(a)}")
    # Privacy is checked when the upload has returned, the first time Kaggle shows the dataset to its owner, and
    # again when Kaggle has processed it. A processing error is reported only after these checks have run.
    confirmed = confirm_dataset_private(a, R, ref, pending_ok=True)
    if confirmed:
        register(private_confirmed=now())

    def on_visible():
        confirm_dataset_private(a, R, ref)                               # strict: alarms unless private
        register(private_confirmed=now())
    state = wait_ready(a, R, ref, on_visible=None if confirmed else on_visible)
    if confirm_dataset_private(a, R, ref, pending_ok=(state == "timeout")) is False:
        register(status="processing")
        die(f"Kaggle is still creating {ref} after {a.wait_min} min. Its page without credentials answers 404 like a made-up "
            f"slug, but its privacy is NOT confirmed yet. Upload nothing else and push no notebook until it is: run the same "
            f"command again (it waits, it does not upload twice), or: python3 kaggle_ft/kaggle_push.py verify-private")
    register(private_confirmed=now(), status={"ready": "ready", "error": "processing_error"}.get(state, "processing"))
    if state == "error":
        die(f"Kaggle reports an error while processing {ref} (it is private: that was checked first). "
            f"Open https://www.kaggle.com/datasets/{ref}")
    say(f"\nIn a notebook that attaches it: {KAGGLE_INPUT}/{slug}/ {mount_note}")
    return ref


DATA_DESCRIPTION = "PRIVATE working copy for the owner's own research. Not for redistribution. Do not make public."


def dataset_source(a, R):
    """Shared start of dataset-create / dataset-version: what is uploaded, as which kind, under which slug.
    The source directory is audited HERE, before upload_location() may copy it to the staging area."""
    src = os.path.abspath(os.path.expanduser(a.dir))
    if not os.path.isdir(src):
        die(f"{src} is not a directory" + (" (run build_bundle.py first)" if src == os.path.abspath(bb.DEFAULT_OUT) else ""))
    kind = resolve_kind(a, src)
    slug = a.slug or SLUGS[kind]
    owner = owner_of(a)
    audit_upload_dir(src, kind)
    folder = upload_location(a, R, src, slug, kind)
    return src, kind, slug, owner, folder, ("zip" if kind == "bundle" else "skip")


def cmd_dataset_create(a, R):
    header(a, R, "dataset-create")
    src, kind, slug, owner, folder, dir_mode = dataset_source(a, R)
    push_dataset(a, R, folder, slug, a.title, a.subtitle, a.description or DATA_DESCRIPTION,
                 dir_mode, kind, owner=owner, scan=src if R.dry else None,
                 mount_note="(gridded/, prompts/, train_manifest.csv, dev_tiles.csv, MANIFEST.json)" if kind == "bundle" else "")


def cmd_dataset_version(a, R):
    header(a, R, "dataset-version")
    src, kind, slug, owner, folder, dir_mode = dataset_source(a, R)
    push_dataset(a, R, folder, slug, a.title, a.subtitle, a.description or DATA_DESCRIPTION,
                 dir_mode, kind, version_notes=a.message, owner=owner, scan=src if R.dry else None)


# -------------------------------------------------------------------------------------------------- weights

HAIDEF_NOTICE = ("HAI-DEF is provided under and subject to the Health AI Developer Foundations Terms of Use found at\n"
                 "https://developers.google.com/health-ai-developer-foundations/terms\n\n"
                 f"This is a private, unmodified copy of {MODEL_ID} (Hugging Face revision {MODEL_REVISION})\n"
                 "kept for its owner's own use. It is not redistributed. Do not make this dataset public.\n")


def git_blob_sha1(path):
    h = hashlib.sha1(b"blob %d\0" % os.path.getsize(path))
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def weights_plan(src, verify_big):
    """Files to upload (top-level, non-hidden) with the hashes Hugging Face published for the pinned revision."""
    if not os.path.isfile(os.path.join(src, "config.json")):
        die(f"{src} does not look like the model directory (no config.json)")
    files, problems = {}, []
    for f in sorted(os.listdir(src)):
        full = os.path.join(src, f)
        if f.startswith(".") or not os.path.isfile(full) or f in ("dataset-metadata.json",):
            continue                                                    # .cache/, .gitattributes, .DS_Store stay local
        e = {"bytes": os.path.getsize(full)}
        mpath = os.path.join(src, ".cache", "huggingface", "download", f + ".metadata")
        if os.path.isfile(mpath):
            with open(mpath) as fh:
                lines = fh.read().split()
            rev, etag = (lines + [None, None])[:2]
            e["hf_revision"] = rev
            if rev != MODEL_REVISION:
                problems.append(f"{f}: downloaded from revision {rev}, not the pinned {MODEL_REVISION}")
            if etag and len(etag) == 64:                                # LFS file: the etag is its sha256
                e["sha256"] = etag
                if e["bytes"] <= 256 << 20 or verify_big:
                    e["verified_locally"] = bb.sha256_file(full) == etag
                else:
                    e["verified_locally"] = None                        # checked on Kaggle by the job (PINNED_SHA256)
            elif etag and len(etag) == 40:                              # small file: the etag is its git blob sha1
                e["git_blob_sha1"] = etag
                e["verified_locally"] = git_blob_sha1(full) == etag
                e["sha256"] = bb.sha256_file(full)
            if e.get("verified_locally") is False:
                problems.append(f"{f}: content does not match the hash Hugging Face published")
        else:
            problems.append(f"{f}: no Hugging Face download record; cannot tie it to the pinned revision")
        files[f] = e
    need = ["config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
            "preprocessor_config.json", "processor_config.json", "chat_template.jinja"]
    idx = os.path.join(src, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx) as fh:
            need += sorted(set(json.load(fh).get("weight_map", {}).values()))
    problems += [f"{f}: required file is missing" for f in need if f not in files]
    return files, problems


def cmd_weights(a, R):
    header(a, R, "weights")
    owner = owner_of(a)
    src = os.path.abspath(a.weights_dir)
    stage = os.path.join(a.state_dir, "weights")
    files, problems = weights_plan(src, a.verify_weights_hash)
    if problems:
        die("the weights directory is not the pinned model:\n  - " + "\n  - ".join(problems))
    total = sum(e["bytes"] for e in files.values())
    say(f"weights: {src}\n  {len(files)} files, {total / 1e9:.2f} GB, all from revision {MODEL_REVISION}")
    for f, e in files.items():
        v = {True: "hash verified here", None: "hash recorded, verified on Kaggle by the job", False: "MISMATCH"}[e.get("verified_locally")]
        say(f"    {e['bytes']:>13,d}  {f}  [{v}]")
    say("  excluded: .cache/ and every dot-file (they stay on this machine)")
    manifest = json.dumps({"model": MODEL_ID, "revision": MODEL_REVISION, "license": "Health AI Developer Foundations Terms of Use",
                           "files": files}, indent=1) + "\n"
    staged_names = sorted(files) + ["WEIGHTS_MANIFEST.json", "NOTICE"]
    if R.dry:
        audit_upload_dir(src, "weights", only=set(files), allow=sorted(files))
        say("local checks passed: no tile name, no label, no secret in the text files")
        say(f"\nplan:\n  would stage hard links (no copy, no extra disk) of those {len(files)} files in {stage}/")
        R.write(os.path.join(stage, "WEIGHTS_MANIFEST.json"), manifest[:600] + ("   ...\n" if len(manifest) > 600 else ""), "WEIGHTS_MANIFEST.json (first lines)")
        R.write(os.path.join(stage, "NOTICE"), HAIDEF_NOTICE, "NOTICE")
        ref = f"{owner}/{check_slug(a.slug, a)}"
        ignore = [x for p in IGNORE for x in ("--ignore-patterns", p)]
        R.kaggle("datasets", "status", ref, note="must FAIL (404): the dataset must not exist yet")
        R.write(os.path.join(stage, "dataset-metadata.json"), dataset_metadata(owner, a.slug, a.slug, WEIGHTS_SUBTITLE, WEIGHTS_DESCRIPTION), "dataset-metadata.json")
        R.kaggle("datasets", "create", "-p", stage, "-r", "skip", *ignore,
                 note=f"private: no -u/--public. {total / 1e9:.1f} GB; resumable: re-run the same command after an interruption")
        wait_ready(a, R, ref)
        confirm_dataset_private(a, R, ref)
        say(f"  would record {ref} in {registry_path(a)}\n\nIn a notebook that attaches it: {KAGGLE_INPUT}/{a.slug}/  (config.json, model-0000*-of-00002.safetensors, tokenizer files)")
        return
    os.makedirs(stage, exist_ok=True)
    for f in sorted(os.listdir(stage)):                                  # the stage holds only what is uploaded
        if f not in files and f not in ("WEIGHTS_MANIFEST.json", "NOTICE", "dataset-metadata.json", ".DS_Store"):
            die(f"unexpected file in the staging directory: {os.path.join(stage, f)} (remove it and re-run)")
    for f in files:
        s, d = os.path.join(src, f), os.path.join(stage, f)
        if os.path.exists(d) and os.path.samefile(s, d):
            continue
        if os.path.lexists(d):
            os.remove(d)
        try:
            os.link(s, d)
        except OSError:
            os.symlink(s, d)
    with open(os.path.join(stage, "WEIGHTS_MANIFEST.json"), "w") as fh:
        fh.write(manifest)
    with open(os.path.join(stage, "NOTICE"), "w") as fh:
        fh.write(HAIDEF_NOTICE)
    say(f"staged (hard links) in {stage}")
    push_dataset(a, R, stage, a.slug, a.slug, WEIGHTS_SUBTITLE, WEIGHTS_DESCRIPTION, "skip", "weights",
                 mount_note="(config.json, model-0000*-of-00002.safetensors, tokenizer files)", owner=owner,
                 allow=staged_names)


WEIGHTS_SUBTITLE = "Private model copy, owner only, not redistributed"
WEIGHTS_DESCRIPTION = (f"PRIVATE, unmodified copy of {MODEL_ID} (Hugging Face revision {MODEL_REVISION}) for the owner's own use "
                       "under the Health AI Developer Foundations Terms of Use "
                       "(https://developers.google.com/health-ai-developer-foundations/terms). Not redistributed. Do not make public.")


# -------------------------------------------------------------------------------------------------- kernels

def expand(value, owner):
    """{bundle} {weights} {jobs} {adapter} {wheels} {train} {infer} -> /kaggle/input/<default slug>; {input:slug} -> /kaggle/input/slug."""
    def sub(m):
        key = m.group(1)
        if key.startswith("input:"):
            return f"{KAGGLE_INPUT}/{key[6:]}"
        if key not in SLUGS:
            die(f"unknown placeholder {{{key}}} (known: {', '.join('{' + k + '}' for k in SLUGS)}, {{input:<slug>}})")
        return f"{KAGGLE_INPUT}/{SLUGS[key]}"
    return re.sub(r"\{([a-z]+(?::[a-z0-9-]+)?)\}", sub, value)


def staged_script(src_text, src_name, env, argv, pip=None):
    """The script as pushed: the source plus a short header (after the docstring / __future__ imports) that records
    where it came from, lists /kaggle/input in the log, optionally pip-installs packages, and sets environment
    variables / command-line arguments. A script kernel is started without arguments, so this is how a job gets
    its paths. The whole of sys.argv is replaced, argv[0] included (it becomes the source file's name): a script
    that looks at argv[0] to detect a notebook kernel must not mistake the injected arguments for the kernel's
    own. Processes the script starts itself (e.g. one worker per GPU) inherit KAGGLE_PUSH_STARTED and are left
    alone: their own command line is kept and nothing is installed or listed twice."""
    tree = ast.parse(src_text)
    after, body, i = 0, tree.body, 0
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        after, i = body[0].end_lineno, 1
    while i < len(body) and isinstance(body[i], ast.ImportFrom) and body[i].module == "__future__":
        after, i = body[i].end_lineno, i + 1
    lines = src_text.splitlines(keepends=True)
    if after == 0 and lines and lines[0].startswith("#!"):
        after = 1
    inj = ["\n", "# ---- injected by kaggle_ft/kaggle_push.py kernel-push; not part of the source file ----\n",
           f"# source: {src_name}  sha256 {sha256_text(src_text)}\n",
           "import os as _kp_os, sys as _kp_sys\n",
           "_kp_top = not _kp_os.environ.get('KAGGLE_PUSH_STARTED')   # False in processes this script starts itself\n",
           "_kp_os.environ['KAGGLE_PUSH_STARTED'] = '1'\n"]
    if env:
        inj.append(f"_kp_os.environ.update({dict(sorted(env.items()))!r})\n")
    if argv is not None:
        inj += ["if _kp_top and (len(_kp_sys.argv) <= 1 or 'ipykernel' in _kp_sys.argv[0] or _kp_sys.argv[1:2] == ['-f']):\n",
                f"    _kp_sys.argv = [{src_name!r}] + {list(argv)!r}\n",
                "    print('[kaggle_push] arguments', _kp_sys.argv[1:], flush=True)\n"]
    inj += ["if _kp_top:\n",
            "    try:\n",
            f"        for _kp_d in sorted(_kp_os.listdir({KAGGLE_INPUT!r})):\n",
            f"            print('[kaggle_push] input', _kp_d, sorted(_kp_os.listdir(_kp_os.path.join({KAGGLE_INPUT!r}, _kp_d)))[:12], flush=True)\n",
            "    except OSError as _kp_e:\n",
            "        print('[kaggle_push] cannot list the input directory:', _kp_e, flush=True)\n"]
    if pip:
        inj += ["    import subprocess as _kp_sp\n",
                f"    _kp_pip = [_kp_sys.executable, '-m', 'pip', 'install', '-q'] + {list(pip)!r}\n",
                "    print('[kaggle_push] ' + ' '.join(_kp_pip), flush=True)\n",
                "    _kp_sp.check_call(_kp_pip)\n"]
    inj += ["# ---- end of injected header ----\n", "\n"]
    out = "".join(lines[:after]) + ("" if not lines[:after] or lines[after - 1].endswith("\n") else "\n") + "".join(inj) + "".join(lines[after:])
    ast.parse(out)
    return out, "".join(inj)


def kernel_metadata(owner, slug, code_file, datasets, kernels, internet, accelerator, gpu=True, pinning=None):
    md = {"id": f"{owner}/{slug}", "title": slug, "code_file": code_file, "language": "python", "kernel_type": "script",
          "is_private": "true", "enable_gpu": "true" if gpu else "false", "enable_tpu": "false",
          "enable_internet": "true" if internet else "false",
          "dataset_sources": datasets, "competition_sources": [], "kernel_sources": kernels, "model_sources": []}
    if gpu and accelerator != "none":
        md["machine_shape"] = accelerator
    if pinning:
        md["docker_image_pinning_type"] = pinning
    assert md["is_private"] == "true" and md["kernel_type"] == "script"
    return json.dumps(md, indent=2) + "\n"


def cmd_kernel_push(a, R):
    header(a, R, "kernel-push")
    script = os.path.abspath(a.script)
    if not os.path.isfile(script) or not script.endswith(".py"):
        die(f"{script} is not a .py file")
    if a.accelerator in RETIRED_ACCELERATORS:
        die(f"--accelerator {a.accelerator}: {RETIRED_ACCELERATORS[a.accelerator]}. Use NvidiaTeslaT4 (GPU T4 x2).")
    owner = owner_of(a)
    slug = check_slug(a.slug, a)
    ref = f"{owner}/{slug}"
    datasets = [to_ref(owner, d, a) for d in a.dataset]
    kernels = [to_ref(owner, k, a) for k in a.kernel_source]
    env = {}
    for kv in a.env:
        if "=" not in kv or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", kv):
            die(f"--env expects NAME=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        if SECRET_NAME.search(k):
            die(f"--env {k}: a variable with such a name looks like a credential. Its value would be written into the "
                "pushed script and into registry.json. No token goes to Kaggle with this tool: attach data as private "
                "datasets instead.")
        env[k] = expand(v, owner)
    if a.enable_internet == "true" and not a.i_accept_internet:
        die("--enable-internet true also needs --i-accept-internet. With internet on, the script and every package pip "
            "installs can reach the network while the tiles and the weights are mounted. The offline route is a "
            "private wheels dataset: --dataset wheels --pip-install '...' --pip-find-links {wheels}")
    argv = [expand(x, owner) for x in shlex.split(a.args)] if a.args is not None else None
    pip = None
    if a.pip_install:
        pkgs = shlex.split(a.pip_install)
        if any(x.startswith("-") for x in pkgs):
            die("--pip-install takes package requirements only (e.g. 'peft>=0.13.0 bitsandbytes'), no pip options")
        if a.pip_find_links:
            pip = ["--no-index", "--find-links", expand(a.pip_find_links, owner)] + pkgs
        elif a.enable_internet == "true":
            pip = pkgs
        else:
            die("--pip-install needs either --enable-internet true or --pip-find-links {input:<your private wheels dataset>}")
    with open(script, encoding="utf-8") as fh:
        src = fh.read()
    if a.verbatim:
        if env or argv is not None or pip:
            die("--verbatim cannot be combined with --args / --env / --pip-install")
        text, inj = src, ""
    else:
        text, inj = staged_script(src, os.path.basename(script), env, argv, pip)
    # local tripwires on exactly what would be sent
    problems = bb.audit_text(os.path.basename(script), text, bb.load_truth())
    problems += [f"contains a secret ({h})" for h in secret_hits(text)]
    if problems:
        print(f"LEAK TRIPWIRE: {script} is NOT pushed.", file=sys.stderr)
        for p in problems:
            print("  - " + p, file=sys.stderr)
        sys.exit(EXIT_LEAK)
    say(f"script: {script}  ({len(src.splitlines())} lines, sha256 {sha256_text(src)[:16]}; no tile label, no test tile, no secret)")
    if inj:
        say("header inserted into the pushed copy (after the docstring):\n" + "".join("      " + ln for ln in inj.strip("\n").splitlines(keepends=True)))
    stage = os.path.join(a.state_dir, "kernels", slug)
    meta = kernel_metadata(owner, slug, os.path.basename(script), datasets, kernels, a.enable_internet == "true",
                           a.accelerator, gpu=a.enable_gpu == "true", pinning=a.docker_image_pinning)
    if a.enable_internet == "true":
        say("NOTE: internet is ON for this run (--i-accept-internet): the notebook stays private, but the script and "
            "pip can reach the network while the private data is mounted.")
    say("" if not R.dry else "\nplan:")
    # The notebook itself, BEFORE anything is sent: an existing notebook with this slug must already be private.
    state, m = kernel_privacy(a, R, ref)
    if state == "PUBLIC":
        alarm_kernel(ref, "it already exists and is_private is not true; nothing was pushed")
    if state != "private":
        mine = list_mine(R, "kernels")
        if mine is not None:
            if ref in mine:
                die(f"{ref} exists on your account but its metadata cannot be read ({state}: "
                    f"{str(m.get('detail', ''))[:120]}). Nothing was pushed; try again.")
            say(f"{ref} does not exist yet: this push creates it (private).")
    elif bool(m.get("enable_internet")) and a.enable_internet != "true":
        say(f"note: the existing notebook {ref} has internet ON; this push sets it OFF, and that is checked afterwards.")
    for d in datasets:                                                   # every attached dataset of ours must be private
        if d.split("/")[0] == owner:
            state, detail = dataset_privacy(a, R, d)
            if state in ("PUBLIC", "SHARED"):
                alarm_dataset(d, detail)
            if state in ("missing", "unknown"):
                die(f"dataset source {d} is not readable ({state}: {detail}). Create it first.")
    for k in kernels:
        if k.split("/")[0] == owner:
            state, m = kernel_privacy(a, R, k)
            if state == "PUBLIC":
                alarm_kernel(k, "is_private is not true")
            if state in ("missing", "unknown"):
                die(f"kernel source {k} is not readable ({state}). Push and finish it first.")
    if R.dry:
        say(f"  would copy the script (with the header above) to {os.path.join(stage, os.path.basename(script))}")
    else:
        if os.path.isdir(stage):
            for f in os.listdir(stage):                                  # the stage holds one script and its metadata
                if f.endswith(".py") or f == "kernel-metadata.json":
                    os.remove(os.path.join(stage, f))
        os.makedirs(stage, exist_ok=True)
        with open(os.path.join(stage, os.path.basename(script)), "w", encoding="utf-8") as fh:
            fh.write(text)
    R.write(os.path.join(stage, "kernel-metadata.json"), meta, "kernel-metadata.json")

    def register(**fields):
        """Record the notebook BEFORE the push, so that verify-private looks at it whatever happens afterwards."""
        if R.dry:
            return
        reg_ = load_registry(a)
        e_ = reg_["kernels"].setdefault(ref, {"created": now()})
        e_.update(fields)
        save_registry(a, reg_)

    register(script=script, script_sha256=sha256_text(src), pushed_sha256=sha256_text(text), dataset_sources=datasets,
             kernel_sources=kernels, enable_internet=a.enable_internet == "true", accelerator=a.accelerator, args=argv,
             env=env, pip=pip, status="pushing", push_started=now(), private_confirmed=None)
    t = ["-t", str(a.timeout)] if a.timeout else []
    rc, out = R.kaggle("kernels", "push", "-p", stage, *t, tee=True,
                       note="creates the private notebook (or a new version of it) and STARTS the run: this spends GPU quota")
    if rc is not None:
        if rc != 0 or "Kernel push error" in out or "successfully pushed" not in out:
            register(status="push_failed", push_failed=now())
            state, m = kernel_privacy(a, R, ref)                         # the server may have taken it all the same
            if state == "PUBLIC":
                alarm_kernel(ref, "is_private is not true")
            die(f"the push did not succeed (exit code {rc}). Kaggle's answer about {ref} right now: {state}. It stays in "
                "the registry, so verify-private keeps checking it.")
        if "not valid" in out:
            say("WARNING: Kaggle rejected some sources (see above). The run will not find them: fix and push again.")
        register(status="pushed", last_push=now())
    state, m = kernel_privacy(a, R, ref)
    for _ in range(6):                                                   # a brand-new notebook can take a moment to be readable
        if state in ("private", "PUBLIC", "dry"):
            break
        time.sleep(getattr(a, "retry_seconds", 10))
        state, m = kernel_privacy(a, R, ref)
    if state != "dry":
        if state != "private":
            register(status=f"ALARM: Kaggle says {state}")
            alarm_kernel(ref, f"{state}: is_private={m.get('is_private')!r}")
        if m.get("enable_internet") is not False and a.enable_internet != "true":
            # Treated like a privacy failure: the run has started with the tiles and the weights mounted.
            # Only an explicit `false` from Kaggle counts as off; a missing value is not taken on trust.
            register(status="ALARM: internet ON although OFF was requested")
            privacy_alarm([f"Kaggle reports internet ON for {ref} although OFF was requested: stop the run on kaggle.com now."
                           if m.get("enable_internet") else
                           f"Kaggle did not report the internet setting of {ref} (enable_internet={m.get('enable_internet')!r}), "
                           "so internet OFF is not confirmed: check it on kaggle.com now and stop the run if it is on.",
                           f"  1. Open https://www.kaggle.com/code/{ref} and stop / cancel the running session.",
                           "  2. In the notebook editor: Settings -> Internet -> off (or delete the notebook).",
                           "  3. Then run: python3 kaggle_ft/kaggle_push.py verify-private",
                           "The notebook is private, but it was NOT recorded as confirmed. Nothing further was done."])
        say(f"PRIVATE confirmed by Kaggle: {ref} (is_private=True, enable_internet={m.get('enable_internet')}, "
            f"enable_gpu={m.get('enable_gpu')}, machine_shape={m.get('machine_shape')!r})")
        if not bool(m.get("enable_internet")) and a.enable_internet == "true":
            say("note: internet was requested, but Kaggle reports it OFF (phone not verified?); pip from the network will fail.")
        missing = sorted(set(datasets) - set(m.get("dataset_sources") or []))
        if missing:
            say(f"WARNING: Kaggle did not attach {missing}")
        register(private_confirmed=now(), status="running or finished: see kernel-status")
    else:
        say(f"  would record {ref} in {registry_path(a)} (before the push, with status 'pushing')")
    say(f"\nnext: kaggle_push.py kernel-status --slug {slug} --wait    then: kaggle_push.py kernel-output --slug {slug}")


def kernel_state(out):
    low = out.lower()
    for word, state in (("complete", "complete"), ("error", "error"), ("cancel", "cancelled"), ("running", "running"),
                        ("queued", "queued"), ("new", "queued")):
        if re.search(r'status "[^"]*' + word, low):
            return state
    return "unknown"


def cmd_kernel_status(a, R):
    header(a, R, "kernel-status")
    owner = owner_of(a)
    ref = to_ref(owner, a.slug, a)
    if R.dry:
        R.kaggle("kernels", "status", ref, note=("repeated every %d s until the status is complete / error / cancelled" % a.interval) if a.wait else
                 "prints: <ref> has status \"...\"")
        return
    t0 = time.time()
    while True:
        rc, out = R.kaggle("kernels", "status", ref)
        say("  " + out.strip().replace("\n", "\n  "))
        state = kernel_state(out) if rc == 0 else "unknown"
        if state == "complete":
            say("  NOTE: 'complete' only means the script exited with status 0. train_lora.py and infer_jobs.py also exit 0\n"
                "  after a pause or a failure that left output worth keeping. Success is: train_summary.json \"status\":\n"
                f"  \"complete\", or <answers>.status.json \"status\": \"complete\". kernel-output --slug {ref.split('/')[1]} prints them.")
        if not a.wait or state in ("complete", "error", "cancelled"):
            sys.exit(0 if (state == "complete" or (not a.wait and rc == 0)) else 1)
        if time.time() - t0 > a.max_wait_min * 60:
            die(f"still {state} after {a.max_wait_min} min; run the command again to keep waiting")
        time.sleep(a.interval)


def cmd_kernel_output(a, R):
    header(a, R, "kernel-output")
    owner = owner_of(a)
    ref = to_ref(owner, a.slug, a)
    dest = os.path.abspath(a.dest or os.path.join(OUT_DIR, ref.split("/")[1]))
    if bb.in_synced_folder(dest):
        die(f"{dest} is inside a folder macOS may sync to iCloud; adapters and model outputs stay out of it")
    extra = (["-o"] if a.force else []) + (["--file-pattern", a.file_pattern] if a.file_pattern else [])
    if R.dry:
        say(f"  would create {dest}/")
    else:
        os.makedirs(dest, exist_ok=True)
    rc, out = R.kaggle("kernels", "output", ref, "-p", dest, *extra, tee=True,
                       note="files already downloaded and unchanged are skipped; the run log is saved as <slug>.log")
    if rc is None:
        return
    if rc != 0:
        die(f"download failed (exit code {rc})")
    files = bb.bundle_files(dest)
    say(f"\n{len(files)} files in {dest}:")
    for f in files[:60]:
        say(f"  {os.path.getsize(os.path.join(dest, f)):>13,d}  {f}")
    if len(files) > 60:
        say(f"  ... and {len(files) - 60} more")
    for line in run_outcomes(dest, files):
        say(line)


def run_outcomes(dest, files):
    """What the downloaded run itself says about how it ended. A notebook that Kaggle shows as 'complete' may be a
    paused or failed run that exited 0 on purpose so that its output was kept."""
    lines = []

    def load(rel):
        try:
            with open(os.path.join(dest, rel)) as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    for rel in files:
        base = os.path.basename(rel)
        if base == "train_summary.json":
            s = load(rel)
            ok = s.get("status") == "complete"
            lines.append(f"  {'OK  ' if ok else 'NOT FINISHED'}  {rel}: status={s.get('status')!r}, protocol_run={s.get('protocol_run')!r}, "
                         f"chosen_epoch={s.get('chosen_epoch')!r}, epochs {s.get('epochs_completed')!r}/{s.get('epochs_planned')!r}"
                         + ("" if ok else "  -> push again with the earlier output attached to resume (README, 'A second training session')"))
        elif base.endswith(".status.json"):
            s = load(rel)
            ok = s.get("status") == "complete"
            lines.append(f"  {'OK  ' if ok else 'NOT FINISHED'}  {rel}: status={s.get('status')!r}, {s.get('n_done')!r}/{s.get('n_jobs')!r} jobs"
                         + ("" if ok else f"  ({s.get('message')})"))
        elif base == "FAILED.txt":
            lines.append(f"  FAILED        {rel} exists: the run stopped with an error (read it)")
    if lines:
        lines.insert(0, "\noutcome reported by the run itself (kernel-status 'complete' is NOT success by itself):")
    return lines


def cmd_dataset_status(a, R):
    header(a, R, "dataset-status")
    owner = owner_of(a)
    ref = to_ref(owner, a.slug, a)
    rc, out = R.kaggle("datasets", "status", ref, note="prints: ready (or the processing state)")
    if rc is not None:
        say("  " + out.strip())
    state, detail = dataset_privacy(a, R, ref)
    if state in ("PUBLIC", "SHARED"):
        alarm_dataset(ref, detail)
    if state != "dry":
        say(f"  privacy: {state} ({detail})")


def cmd_quota(a, R):
    header(a, R, "quota")
    rc, out = R.kaggle("quota", note="weekly GPU / TPU hours: used, remaining, total, reset time")
    if rc is not None:
        say(out.strip())
        sys.exit(rc)


# ---------------------------------------------------------------------------------------------------- check

def cmd_check(a, R):
    header(a, R, "check")
    c = read_credentials()
    say(f"credentials file: {c['path']}")
    if c["exists"]:
        say(f"  permissions: {c['mode']:03o} " + ("(owner only: OK)" if c["mode_ok"] else "(NOT OK: readable by others)"))
        say(f"  username: {c['username'] or 'MISSING'}")
        say("  key: " + ("present (never shown)" if c["has_key"] else "MISSING"))
    else:
        say("  MISSING")
    for n in c["notes"]:
        say("  note: " + n)
    if c["problems"]:
        say("\nNOT READY:\n  - " + "\n  - ".join(c["problems"]))
        say("\nOne-time setup:\n"
            "  1. Sign in at https://www.kaggle.com/settings -> API -> 'Legacy API Credentials' -> 'Create Legacy API Key'.\n"
            "     The browser downloads kaggle.json.\n"
            f"  2. mkdir -p {config_dir()} && mv ~/Downloads/kaggle.json {c['path']} && chmod 600 {c['path']}\n"
            "  3. On the same settings page, verify your phone number (needed for GPU and for internet in notebooks).\n"
            "  Never paste the key into a chat, a script or a notebook.")
    say("\nnetwork checks" + (" (skipped in a dry run; these are the commands):" if R.dry else ":"))
    rc, out = R.kaggle("config", "view", note="the CLI must log in as the same username")
    rc2, out2 = R.kaggle("quota", note="proves the token works; shows the weekly GPU hours left")
    if c["problems"]:
        sys.exit(EXIT_USAGE)
    if rc is None:
        say("\nlocal checks passed (network checks not run)")
        return
    m = re.search(r"-\s*username\s*:\s*(\S+)", out)
    seen = m.group(1) if m else None
    if rc != 0 or seen in (None, "None"):
        die("the Kaggle CLI could not log in with these credentials")
    if c.get("kind") == "access_token" and c["username"] is None and USER_RE.match(seen or ""):
        os.makedirs(os.path.dirname(_account_cache()), exist_ok=True)
        with open(_account_cache(), "w") as fh:
            json.dump({"username": seen, "source": "kaggle config view (access_token)"}, fh)
        c["username"] = seen
        say(f"  cached the account name for later commands: {seen}")
    if seen != c["username"]:
        die(f"the Kaggle CLI logs in as {seen}, but {c['path']} says {c['username']}: remove the other credential "
            "(~/.kaggle/access_token, `kaggle auth` login or KAGGLE_* environment variables) or fix kaggle.json")
    say(f"  the CLI logs in as: {seen}")
    say("  " + out2.strip().replace("\n", "\n  "))
    if rc2 != 0:
        die("`kaggle quota` failed: the token did not work, or the account has no accelerator access yet (verify your phone)")
    say("\ncheck passed")


# ------------------------------------------------------------------------------------------- verify-private

def list_mine(R, group):
    """All refs of the account's own datasets or notebooks (private ones included)."""
    refs, token = [], None
    for _ in range(50):
        args = [group, "list", "--mine", "--csv", "--page-size", "100"] + (["--page-token", token] if token else [])
        rc, out = R.kaggle(*args, note="lists your own " + group + ", private ones included; every ref starting with "
                                       f"{PREFIX!r} is checked" if not token else "")
        if rc is None:
            return None
        if rc != 0:
            die(f"`kaggle {group} list --mine` failed:\n{out.strip()[-400:]}")
        m = re.search(r"Next Page Token = (\S+)", out)
        body = "\n".join(ln for ln in out.splitlines() if not ln.startswith("Next Page Token") and not ln.startswith("Warning"))
        rows = list(csv.reader(io.StringIO(body)))
        if rows and rows[0] and rows[0][0] == "ref":
            refs += [r[0] for r in rows[1:] if r and "/" in r[0]]
        token = m.group(1) if m else None
        if not token:
            break
    return refs


def cmd_verify_private(a, R):
    header(a, R, "verify-private")
    owner = owner_of(a)
    reg = load_registry(a)
    targets = {"dataset": set(reg["datasets"]), "kernel": set(reg["kernels"])}
    say(f"registry: {registry_path(a)} ({len(reg['datasets'])} datasets, {len(reg['kernels'])} notebooks)")
    if R.dry:
        say("\nplan:")
    listed = {"dataset": set(), "kernel": set()}
    for kind, group in (("dataset", "datasets"), ("kernel", "kernels")):
        mine = list_mine(R, group)
        if mine is None:                                                 # dry run: show the default names
            targets[kind] |= {f"{owner}/{SLUGS[k]}" for k in (("bundle", "weights") if kind == "dataset" else ("train", "infer"))}
        else:
            listed[kind] = set(mine)
            targets[kind] |= {r for r in mine if a.all or r.split("/", 1)[1].startswith(PREFIX)}
    control = None
    if not a.no_anon_probe:
        control_url = page_url("dataset", f"{owner}/{PREFIX}no-such-thing-{int(time.time()) % 100000}")
        if R.dry:
            say("  anonymous GET (no credentials, no cookies, no redirects) of each page below; anything served although a\n"
                f"  made-up slug is not ({control_url}) counts as VISIBLE:")
        else:
            control = anon_status(control_url)
    rows, bad, pending = [], [], []
    for kind in ("dataset", "kernel"):
        for ref in sorted(targets[kind]):
            internet_bad = False
            if kind == "dataset":
                state, detail = dataset_privacy(a, R, ref)
            else:
                state, m = kernel_privacy(a, R, ref)
                detail = f"is_private={m.get('is_private')!r}" if state in ("private", "PUBLIC") else str(m.get("detail", ""))[:80]
                if state == "private":
                    # internet may be ON only where the registry says this tool pushed it that way
                    internet_bad = m.get("enable_internet") is not False and not reg["kernels"].get(ref, {}).get("enable_internet")
                    detail += ", internet " + (("ON, expected OFF" if m.get("enable_internet") else "not reported, expected OFF")
                                               if internet_bad else ("on" if m.get("enable_internet") else "off"))
            if R.dry:
                if not a.no_anon_probe:
                    say(f"  GET {page_url(kind, ref)}")
                continue
            probe, pdetail = ("skipped", "") if a.no_anon_probe else anon_probe(kind, ref, control)
            ok = state == "private" and probe != "VISIBLE" and not internet_bad
            e = reg["datasets"].get(ref, {}) if kind == "dataset" else {}
            awaited = (kind == "dataset" and e.get("status") in ("uploading", "uploaded", "processing")
                       and not e.get("private_confirmed") and ref not in listed[kind])
            if awaited and state in ("unknown", "missing"):
                # An upload of this tool that Kaggle has never shown to its owner. It is "still being created" only
                # on positive evidence (the page without credentials is 404 like a made-up slug); it is never "gone"
                # and never counted as private.
                hidden, seen = (False, "anonymous probe skipped") if a.no_anon_probe else anon_pending_evidence(ref)
                if hidden or (a.no_anon_probe and state == "missing"):
                    state = "PENDING"
                    pending.append(ref)
                    rows.append((kind, ref, state, "Kaggle has not created it yet; NOT confirmed", probe, pdetail, False))
                    continue
            elif state == "missing" and ref not in listed[kind] and probe != "VISIBLE":
                state, ok = "gone", True                                 # Kaggle says it does not exist, and it is not listed
            rows.append((kind, ref, state, detail, probe, pdetail, ok))
            if not ok:
                bad.append((kind, ref, state + ("; internet is not confirmed OFF although the registry expects OFF" if internet_bad else ""), probe))
    if R.dry:
        say("  exit code 0 only if every one is private, none is visible anonymously and no notebook has internet on\n"
            "  unless it was pushed that way; otherwise exit code 3. An unanswered metadata request counts as a failure.")
        return
    if not rows:
        say("nothing to verify: no dataset or notebook of this tool exists on the account yet")
        return
    say("")
    for kind, ref, state, detail, probe, pdetail, ok in rows:
        say(f"  {'OK  ' if ok else ('WAIT' if state == 'PENDING' else 'FAIL')}  {kind:<8} {ref:<48} {state:<8} {detail:<42} anonymous view: {probe} {pdetail}")
    stamp = now()
    for kind, ref, state, detail, probe, pdetail, ok in rows:
        e = reg["datasets" if kind == "dataset" else "kernels"].get(ref)
        if e is not None:
            e["last_verify"] = {"at": stamp, "state": state, "anonymous": probe}
            if ok and state == "private":
                e["private_confirmed"] = stamp
    save_registry(a, reg)
    if bad:
        privacy_alarm([f"{kind} {ref}: Kaggle says {state}; anonymous view: {probe}" for kind, ref, state, probe in bad]
                      + ["Open each one on kaggle.com and make it private to you alone (datasets: Settings -> Visibility = Private,",
                         "no collaborators; notebooks: Share -> Private), or delete it. Then run verify-private again.",
                         "('unknown' means Kaggle did not answer the metadata request. It is never counted as private: run",
                         " verify-private again. If the entry is an upload that failed before Kaggle created anything, and",
                         " kaggle.com shows no such dataset, remove it from registry.json by hand.)",
                         "(internet ON: stop the session on kaggle.com, switch Internet off in the editor's Settings, verify again.)"])
    if pending:
        die("Kaggle is still creating: " + ", ".join(pending) + ". Their privacy is NOT confirmed yet (this is not an alarm: "
            "nothing is visible, and Kaggle reported nothing as public). Upload nothing else and push no notebook that uses them; "
            "run verify-private again in a few minutes. (If an upload failed before Kaggle created anything and kaggle.com "
            "shows no such dataset, remove its entry from registry.json by hand.)")
    n_gone = sum(1 for r in rows if r[2] == "gone")
    say(f"\nall {len(rows) - n_gone} private" + ("" if a.no_anon_probe else " and not visible without credentials")
        + (f" ({n_gone} more no longer exist on Kaggle)" if n_gone else "")
        + "; no notebook has internet on that was not pushed that way")


# ------------------------------------------------------------------------------------------------------ cli

def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dry-run", action="store_true", help="print the kaggle commands and metadata files; no network, no writes")
    common.add_argument("--owner", help="Kaggle username; must be the account in ~/.kaggle/kaggle.json, which is where it is read from if omitted")
    common.add_argument("--kaggle-bin", default=KAGGLE_BIN, help="Kaggle CLI (env KAGGLE_BIN)")
    common.add_argument("--state-dir", default=STATE_DIR, help="registry + staging directory (env MHIST_KAGGLE_STATE)")
    common.add_argument("--any-slug", action="store_true", help=f"allow a slug that does not start with {PREFIX!r}")
    common.add_argument("--allow-foreign", action="store_true", help="allow attaching a dataset / notebook of another account")
    common.add_argument("--retry-seconds", type=float, default=10, help=argparse.SUPPRESS)   # waits between metadata retries
    common.add_argument("--poll-seconds", type=float, default=15, help=argparse.SUPPRESS)    # waits between status polls (tests use 0)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="SUBCOMMAND")

    def add(name, fn, help_):
        p = sub.add_parser(name, parents=[common], help=help_, description=help_, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.set_defaults(fn=fn)
        return p

    add("check", cmd_check, "Verify ~/.kaggle/kaggle.json (exists, owner-only permissions), print the username (never the key), "
                            "then confirm the login and the GPU quota with Kaggle.")

    def dataset_args(p, default_slug):
        p.add_argument("--dir", default=bb.DEFAULT_OUT, help="directory to upload; the default is the data bundle")
        p.add_argument("--kind", choices=UPLOAD_KINDS, default=None,
                       help="what --dir is; required for anything but the data bundle. Only that kind's files are accepted: "
                            "bundle = exactly MANIFEST.json's files; jobs = label-free *.jsonl job files (build_dev_jobs.py); "
                            "adapter = adapter_config.json, adapter_model.safetensors, step4_meta.json [, README.md]; "
                            "wheels = *.whl. There is no kind for arbitrary files")
        p.add_argument("--slug", default=None,
                       help=f"dataset slug (default by kind: {', '.join(k + ' -> ' + SLUGS[k] for k in UPLOAD_KINDS)})")
        p.add_argument("--title", default=None, help="dataset title, 6-50 characters; the slug if omitted")
        p.add_argument("--subtitle", default="Private working data, owner only", help="20-80 characters")
        p.add_argument("--description", default=None)
        p.add_argument("--wait-min", type=float, default=45, help="minutes to wait for Kaggle to finish processing")

    p = add("dataset-create", cmd_dataset_create, "Create a NEW PRIVATE dataset from a directory (default: the data bundle). "
                                                  "Sub-directories are uploaded as zip archives. Verified private afterwards.")
    dataset_args(p, SLUGS["bundle"])
    p = add("dataset-version", cmd_dataset_version, "Upload a new version of an existing private dataset. Privacy is checked before and after.")
    dataset_args(p, SLUGS["bundle"])
    p.add_argument("-m", "--message", required=True, help="version notes")
    p.add_argument("--delete-old-versions", action="store_true", help="ask Kaggle to delete the previous versions")
    p.add_argument("--force", action="store_true", help="upload even if the registry says this content is already there")

    p = add("weights", cmd_weights, f"Upload the local {MODEL_ID} copy as a PRIVATE dataset (hard-link staging, no extra disk; "
                                    ".cache and dot-files excluded). Verified private afterwards.")
    p.add_argument("--weights-dir", default=WEIGHTS_DIR)
    p.add_argument("--slug", default=SLUGS["weights"])
    p.add_argument("--verify-weights-hash", action="store_true", help="also sha256 the two large weight files here (reads 8.6 GB)")
    p.add_argument("--wait-min", type=float, default=180)
    p.set_defaults(force=False, delete_old_versions=False)

    p = add("kernel-push", cmd_kernel_push, "Push a Python script as a PRIVATE GPU notebook and start it. Verified private afterwards.")
    p.add_argument("--script", required=True, help="the .py file to run on Kaggle")
    p.add_argument("--slug", required=True, help=f"notebook slug, e.g. {SLUGS['train']}")
    p.add_argument("--dataset", action="append", default=[], metavar="D",
                   help="dataset to attach: bundle | weights | jobs | adapter | wheels | <slug> | <owner>/<slug>; repeatable")
    p.add_argument("--kernel-source", action="append", default=[], metavar="K",
                   help="notebook whose OUTPUT to attach (e.g. train -> the adapter at /kaggle/input/%s/); repeatable" % SLUGS["train"])
    p.add_argument("--args", default=None, metavar="'...'",
                   help="command line for the script on Kaggle; write it as --args='--flag value ...'. "
                        "{bundle} {weights} {jobs} {adapter} {wheels} {train} {input:<slug>} expand to /kaggle/input/<slug>")
    p.add_argument("--env", action="append", default=[], metavar="NAME=VALUE",
                   help="environment variable for the script; repeatable; same placeholders. Names that look like a "
                        "credential (token, key, secret, password) are refused")
    p.add_argument("--pip-install", default=None, metavar="'PKG ...'",
                   help="packages to pip-install on Kaggle before the script starts, e.g. 'peft>=0.13.0 bitsandbytes'; "
                        "needs --enable-internet true or --pip-find-links")
    p.add_argument("--pip-find-links", default=None, metavar="PATH",
                   help="install offline (--no-index) from wheels in this Kaggle path, e.g. {wheels}")
    p.add_argument("--verbatim", action="store_true", help="push the file unchanged (no header, so no --args / --env / --pip-install)")
    p.add_argument("--enable-internet", choices=["true", "false"], default="false",
                   help="internet in the notebook; 'true' also needs --i-accept-internet")
    p.add_argument("--i-accept-internet", action="store_true",
                   help="required with --enable-internet true: you accept that the script and anything pip installs can "
                        "reach the network while the private tiles and weights are mounted")
    p.add_argument("--enable-gpu", choices=["true", "false"], default="true")
    p.add_argument("--accelerator", default="NvidiaTeslaT4", choices=ACCELERATORS + tuple(RETIRED_ACCELERATORS),
                   help="machine_shape; NvidiaTeslaT4 is 'GPU T4 x2'; none = let Kaggle choose")
    p.add_argument("--timeout", type=int, default=None, help="stop the run after this many seconds (Kaggle's own limit is 12 h)")
    p.add_argument("--docker-image-pinning", choices=["original", "latest"], default=None,
                   help="docker_image_pinning_type: 'original' keeps later versions of this notebook on the image of its first run")

    p = add("kernel-status", cmd_kernel_status, "Show the status of a notebook run; --wait polls until it finishes.")
    p.add_argument("--slug", required=True)
    p.add_argument("--wait", action="store_true")
    p.add_argument("--interval", type=int, default=120, help="seconds between polls")
    p.add_argument("--max-wait-min", type=float, default=720)

    p = add("kernel-output", cmd_kernel_output, f"Download a notebook's output files and log to {OUT_DIR}/<slug>/.")
    p.add_argument("--slug", required=True)
    p.add_argument("--dest", default=None, help="another destination directory (must be outside iCloud-synced folders)")
    p.add_argument("--file-pattern", default=None, help="regex: only matching file names")
    p.add_argument("--force", action="store_true", help="download again even if a file looks up to date")

    p = add("dataset-status", cmd_dataset_status, "Processing status and privacy of one dataset.")
    p.add_argument("--slug", required=True)

    p = add("verify-private", cmd_verify_private, f"Assert that every dataset / notebook in the registry, and every one on the account whose "
                                                  f"slug starts with {PREFIX!r}, is private; also look at each page without credentials.")
    p.add_argument("--all", action="store_true", help="check EVERY dataset and notebook on the account, whatever its name")
    p.add_argument("--no-anon-probe", action="store_true", help="skip the anonymous page check")

    add("quota", cmd_quota, "Show the weekly GPU quota (kaggle quota).")

    a = ap.parse_args(argv)
    a.state_dir = os.path.abspath(os.path.expanduser(a.state_dir))
    if bb.in_synced_folder(a.state_dir):
        die(f"--state-dir {a.state_dir} is inside a folder macOS may sync to iCloud")
    a.fn(a, Runner(a.dry_run, os.path.expanduser(a.kaggle_bin)))


if __name__ == "__main__":
    main()
