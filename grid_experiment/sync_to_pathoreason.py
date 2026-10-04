"""Copy new/changed grid_experiment files into a PathoReason clone, check them, commit, push.

Safe by construction for a PUBLIC repo:
  - never copies MHIST pixels (*.png/*.jpg/*.tif/*.pdf/*.zip), .env, outreach/, paper/,
    runs/logs/, macOS/iCloud duplicates ("* 2.*", *.icloud), or anything git ignores
  - HOLD_BACK: files that would unblind the pending expert review stay local until
    the ratings are back
  - copies only files that are new or whose content differs, and refuses to overwrite
    a file whose last repo commit was made by someone else
  - aborts before committing if any staged file matches an API-key/token pattern or is
    an image

    python3 sync_to_pathoreason.py <clone_dir> "<commit subject>" [body_file] [--dry-run]
"""

import fnmatch
import hashlib
import os
import re
import shutil
import subprocess
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
SKIP_DIRS = {"outreach", "paper", "__pycache__", "clean", "gridded", "images", "logs", ".git", "superseded"}
SKIP_PAT = ["* 2.*", "* 2", "*.icloud", ".env", ".DS_Store", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff",
            "*.pdf", "*.zip", "*.pyc", "*.pt"]
# listed in a local-only file (outreach/ is never pushed) so this public script names nothing
_hb = os.path.join(SRC, "outreach", "HOLD_BACK.txt")
HOLD_BACK = {l.strip() for l in open(_hb) if l.strip() and not l.startswith("#")} if os.path.exists(_hb) else set()
OUR_AUTHOR = "FocusFlow Developer"
SECRET = re.compile(r"sk-or-v1-[A-Za-z0-9]{10,}|csk-[a-z0-9]{20,}|gh[opsu]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}")
CO_AUTHOR = "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"


def git(repo, *a, **kw):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, **kw)


def plan(repo):
    cands = []
    for root, dirs, files in os.walk(SRC):
        rel = os.path.relpath(root, SRC)
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not fnmatch.fnmatch(d, "* 2")]
        if rel.startswith("rating/images"):
            continue
        for f in files:
            if not any(fnmatch.fnmatch(f, p) for p in SKIP_PAT):
                c = os.path.normpath(os.path.join(rel, f))
                if c not in HOLD_BACK:
                    cands.append(c)
    ignored = set(git(repo, "check-ignore", "--stdin",
                      input="\n".join("grid_experiment/" + c for c in cands)).stdout.split("\n"))
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    new, chg, blocked = [], [], []
    for c in cands:
        r = "grid_experiment/" + c
        if r in ignored:
            continue
        dst = os.path.join(repo, r)
        if not os.path.exists(dst):
            new.append(c)
        elif sha(dst) != sha(os.path.join(SRC, c)):
            last = git(repo, "log", "-1", "--format=%an", "--", r).stdout.strip()
            (chg if last in ("", OUR_AUTHOR) else blocked).append(c)
    return new, chg, blocked


def main():
    repo, subject = sys.argv[1], sys.argv[2]
    body = open(sys.argv[3]).read() if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else ""
    dry = "--dry-run" in sys.argv
    if git(repo, "pull", "-q", "--ff-only").returncode != 0:
        sys.exit("git pull --ff-only failed; resolve by hand")
    new, chg, blocked = plan(repo)
    print(f"new {len(new)}, changed {len(chg)}, blocked (last edited by someone else) {len(blocked)}")
    for b in blocked:
        print("  BLOCKED, not copied:", b)
    if dry or not (new or chg):
        return
    for c in new + chg:
        dst = os.path.join(repo, "grid_experiment", c)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(SRC, c), dst)
    git(repo, "add", "-A", "grid_experiment")
    staged = git(repo, "diff", "--cached", "--name-only").stdout.split()
    bad = [f for f in staged if re.search(r"\.(png|jpe?g|tiff?|pdf|zip)$", f, re.I)]
    leaks = [f for f in staged if os.path.isfile(os.path.join(repo, f))
             and SECRET.search(open(os.path.join(repo, f), errors="ignore").read())]
    if bad or leaks:
        git(repo, "reset", "-q")
        sys.exit(f"ABORT: images staged {bad[:5]} / secret pattern in {leaks[:5]}; nothing committed")
    msg = subject + ("\n\n" + body.strip() if body.strip() else "") + "\n\n" + CO_AUTHOR + "\n"
    r = git(repo, "commit", "-q", "-F", "-", input=msg)
    if r.returncode != 0:
        sys.exit("commit failed: " + r.stderr[-300:])
    r = git(repo, "push", "-q", "origin", "HEAD:main")
    if r.returncode != 0:
        sys.exit("push failed: " + r.stderr[-300:])
    print("pushed", git(repo, "log", "--oneline", "-1").stdout.strip(), f"({len(staged)} files)")


if __name__ == "__main__":
    main()
