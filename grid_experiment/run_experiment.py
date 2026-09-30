"""Run gridded tiles through a pathoreason prompt on Cerebras.

OpenAI-compatible endpoint, free-form JSON, parse-and-validate. The raw response
string is logged for every call so key order stays auditable after the fact --
never reconstruct it from the parsed object.

    python3 run_experiment.py --prompt cte_p1

Writes runs/<prompt_id>.jsonl, one record per call, appended as calls complete
so a crash or rate-limit stall does not lose completed work.
"""

import argparse
import base64
import hashlib
import json
import os
import random
import re
import sys
import time

import requests
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(ROOT, "prompts", "pathoreason.yaml")
RENDERED = os.path.join(ROOT, "prompts", "rendered")
MANIFEST = os.path.join(ROOT, "selection_manifest.json")
RUNS = os.path.join(ROOT, "runs")

# Endpoint, model, key variable and pacing are environment-overridable so the
# same pipeline can target another host serving the same weights (e.g. when a
# provider archives a model). Defaults reproduce the original Cerebras setup.
BASE_URL = os.environ.get("PATHO_BASE_URL", "https://api.cerebras.ai/v1").rstrip("/")
MODEL = os.environ.get("PATHO_MODEL", "gemma-4-31b")
KEY_ENV = os.environ.get("PATHO_KEY_ENV", "CEREBRAS_API_KEY")
MAX_TOKENS_PARAM = os.environ.get("PATHO_MAX_TOKENS_PARAM", "max_completion_tokens")
# On a router (OpenRouter) the same model id can be served by several upstream
# providers with different quantization; comparisons are only within-host if the
# upstream is pinned. PATHO_PROVIDER pins it with fallbacks disabled, and the
# provider that actually served each call is logged in meta.provider.
PROVIDER = os.environ.get("PATHO_PROVIDER") or None
# Optional sampling overrides. Unset means the HOST's default applies (earlier runs
# never set these, so Cerebras and Friendli each used their own defaults). When set
# they are sent on every call, recorded in every record, and folded into host_tag so
# outputs under different sampling never share a file.
SAMPLING = {k: cast(os.environ[e]) for k, e, cast in (("top_p", "PATHO_TOP_P", float),
                                                    ("top_k", "PATHO_TOP_K", int),
                                                    ("min_p", "PATHO_MIN_P", float))
            if os.environ.get(e)}
# Optional extra request fields as JSON, e.g. '{"reasoning": {"max_tokens": 0}}' to turn
# thinking off for Gemini 2.5 Flash. Sent on every call, recorded, and folded into host_tag.
EXTRA_BODY = json.loads(os.environ["PATHO_EXTRA_BODY"]) if os.environ.get("PATHO_EXTRA_BODY") else {}
MAX_TOKENS = 1500
FATAL_STATUSES = (401, 402, 403, 404)
# A call whose response is received but unusable (unparseable JSON, label not HP/SSA)
# is recorded as an error and retried, at most MAX_TRIES times per (tile, replicate);
# after that it is "given up" and reported, so one bad tile cannot block a run forever.
MAX_TRIES = 3


def counts_as_try(err):
    return str(err or "").startswith(("parse_failed", "invalid_label"))


def mark_invalid(rec):
    """Stamp a received-but-unusable response as an error so resume and completeness agree."""
    if not rec.get("error") and (rec.get("parsed") or {}).get("label") not in ("HP", "SSA"):
        rec["error"] = ("parse_failed" if not rec.get("parsed")
                        else f"invalid_label {rec['parsed'].get('label')!r}")
    return rec


def host_tag():
    """Filesystem-safe id of the serving endpoint + model (+ pinned provider)."""
    host = re.sub(r"^https?://", "", BASE_URL).split("/")[0]
    host = host.replace("api.", "").split(".")[0]
    tag = f"{host}__{MODEL}" + (f"__{PROVIDER}" if PROVIDER else "")
    if SAMPLING:
        tag += "__" + "-".join(f"{k.replace('_', '')}{v}" for k, v in sorted(SAMPLING.items()))
    if EXTRA_BODY:
        tag += "__x" + hashlib.sha256(json.dumps(EXTRA_BODY, sort_keys=True).encode()).hexdigest()[:6]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", tag)

# Temperature is non-zero deliberately: question 3 asks whether cited cells are
# stable across repeated runs, which is only a meaningful question above greedy
# decoding. Matches generation.temperature in pathoreason.yaml.
TEMPERATURE = 1.0

# Free tier is 5 requests/min. 13s between request starts leaves headroom for
# clock skew against the server's window.
MIN_INTERVAL_S = float(os.environ.get("PATHO_MIN_INTERVAL_S", "13.0"))
MAX_ATTEMPTS = 6
BACKOFF_BASE_S = 20.0

# Repeated for the stability check: spans label, agreement band, and control
# class rather than being an arbitrary three.
REPEAT_TILES = ["MHIST_ccn.png", "MHIST_ddn.png", "MHIST_cxx.png"]
N_REPEATS = 3


def b64_png(path):
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def b64_gridded_tile(tile):
    """Base64 PNG of the gridded tile; renders in memory when no file exists.

    The 20 selected tiles keep using their on-disk renders; the full test
    sweep draws the identical overlay on the fly instead of writing ~1000
    PNGs nothing else reads.
    """
    if tile.get("gridded"):
        return b64_png(os.path.join(ROOT, tile["gridded"]))
    import io

    from PIL import Image

    from grid import draw_grid

    img = draw_grid(Image.open(os.path.join(ROOT, "..", "images", tile["image"])))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def band(votes):
    if votes in (0, 7):
        return "unanimous"
    if votes in (3, 4):
        return "borderline"
    return "strong"


def load_full_test():
    """All 977 test-partition tiles, shaped like selection_manifest entries."""
    import csv

    tiles = []
    for row in csv.DictReader(open(os.path.join(ROOT, "coverage_index.csv"))):
        if row["partition"] != "test":
            continue
        cov = json.load(open(os.path.join(ROOT, "coverage", row["image"].replace(".png", ".json"))))
        tiles.append(
            {
                "image": row["image"],
                "gridded": None,
                "label": row["label"],
                "ssa_votes_out_of_7": int(row["ssa_votes"]),
                "agreement_band": band(int(row["ssa_votes"])),
                "n_empty_cells": int(row["n_empty_cells"]),
                "empty_cells": cov["empty_cells"],
            }
        )
    tiles.sort(key=lambda t: t["image"])
    return tiles


def top_level_key_order(raw):
    """Key order as it appears in the raw string, not as parsed.

    Scans only at brace depth 1 and ignores anything inside a string literal, so
    a key name occurring inside a description does not register.
    """
    depth = 0
    in_str = False
    esc = False
    keys = []
    i = 0
    while i < len(raw):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                # a key is a depth-1 string followed by a colon
                if depth == 1:
                    j = i + 1
                    while j < len(raw) and raw[j] in " \t\r\n":
                        j += 1
                    if j < len(raw) and raw[j] == ":":
                        keys.append(cur)
        else:
            if ch == '"':
                in_str = True
                cur_start = i + 1
                end = cur_start
                # find the closing quote to capture the literal
                k, e2 = cur_start, False
                while k < len(raw):
                    if e2:
                        e2 = False
                    elif raw[k] == "\\":
                        e2 = True
                    elif raw[k] == '"':
                        break
                    k += 1
                cur = raw[cur_start:k]
            elif ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
        i += 1
    return keys


def extract_json(raw):
    """Pull the JSON object out of a possibly fenced / prefixed response."""
    s = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if fence:
        s = fence.group(1).strip()
    start = s.find("{")
    if start < 0:
        return None, "no opening brace"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1], None
    return None, "unbalanced braces"


def validate(obj, spec):
    """Return (list of violations, normalised record). Never raises on bad input."""
    v = []
    vocab = set(spec["feature_vocabulary"])
    cells_ok = set(spec["valid_grid_cells"])
    labels_ok = set(spec["valid_labels"])

    label = obj.get("label")
    if label not in labels_ok:
        v.append(f"label {label!r} not in {sorted(labels_ok)}")

    conf = obj.get("confidence")
    if not isinstance(conf, (int, float)) or isinstance(conf, bool):
        v.append(f"confidence {conf!r} is not a number")
    elif not 0.0 <= float(conf) <= 1.0:
        v.append(f"confidence {conf} outside [0,1]")

    ev = obj.get("evidence")
    norm_ev = []
    if not isinstance(ev, list):
        v.append(f"evidence is {type(ev).__name__}, expected list")
    else:
        for n, item in enumerate(ev):
            if not isinstance(item, dict):
                v.append(f"evidence[{n}] is {type(item).__name__}, expected object")
                continue
            feat = item.get("feature")
            if feat not in vocab:
                v.append(f"evidence[{n}].feature {feat!r} outside vocabulary")
            gc = item.get("grid_cells")
            good_cells, bad_cells = [], []
            if not isinstance(gc, list) or not gc:
                v.append(f"evidence[{n}].grid_cells must be a non-empty list, got {gc!r}")
                gc = []
            for c in gc:
                (good_cells if isinstance(c, str) and c.strip().upper() in cells_ok
                 else bad_cells).append(c)
            for c in bad_cells:
                v.append(f"evidence[{n}].grid_cells contains invalid cell {c!r}")
            if not isinstance(item.get("description"), str) or not item.get("description", "").strip():
                v.append(f"evidence[{n}].description missing or empty")
            norm_ev.append(
                {
                    "feature": feat,
                    "grid_cells_valid": [c.strip().upper() for c in good_cells],
                    "grid_cells_invalid": bad_cells,
                    "description": item.get("description"),
                }
            )
    return v, {"label": label, "confidence": conf, "evidence": norm_ev}


def call(session, prompt_text, img_b64, api_key, temperature=TEMPERATURE):
    """One chat completion with backoff. Returns (raw_text, meta, error)."""
    body = {
        "model": MODEL,
        MAX_TOKENS_PARAM: MAX_TOKENS,
        "temperature": temperature,
        **SAMPLING,
        **EXTRA_BODY,
        **({"provider": {"order": [PROVIDER], "allow_fallbacks": False}} if PROVIDER else {}),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    attempts = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        t0 = time.time()
        try:
            r = session.post(
                f"{BASE_URL}/chat/completions", headers=headers, json=body, timeout=180
            )
        except requests.RequestException as e:
            attempts.append({"attempt": attempt, "error": f"{type(e).__name__}: {e}"})
            time.sleep(BACKOFF_BASE_S * (2 ** (attempt - 1)) + random.uniform(0, 3))
            continue

        attempts.append(
            {"attempt": attempt, "status": r.status_code, "elapsed_s": round(time.time() - t0, 2)}
        )
        if r.status_code == 200:
            try:
                d = r.json()
                raw = d["choices"][0]["message"].get("content") or ""
            except (ValueError, KeyError, IndexError, TypeError):
                # some routers return an error object with HTTP 200
                attempts[-1]["error_body"] = r.text[:300]
                try:
                    code = int((r.json().get("error") or {}).get("code"))
                except (ValueError, TypeError, AttributeError):
                    code = None
                if code in FATAL_STATUSES:
                    sys.exit(f"FATAL error code {code} in HTTP-200 body from {BASE_URL}: {r.text[:300]}")
                if sum(1 for a in attempts if "error_body" in a) >= 3:
                    return None, {"attempts": attempts}, "error_body_200"
                time.sleep(BACKOFF_BASE_S * (2 ** (attempt - 1)) + random.uniform(0, 3))
                continue
            return raw, {
                "usage": d.get("usage"),
                "finish_reason": d["choices"][0].get("finish_reason"),
                "model": d.get("model"),
                "provider": d.get("provider"),
                "system_fingerprint": d.get("system_fingerprint"),
                "attempts": attempts,
            }, None
        if r.status_code in FATAL_STATUSES:
            # auth, billing, or model gone: every further call would fail the same
            # way, so stop instead of writing one error record per tile
            sys.exit(f"FATAL HTTP {r.status_code} from {BASE_URL} ({MODEL}): {r.text[:300]}")
        if r.status_code in (429, 500, 502, 503, 504):
            wait = BACKOFF_BASE_S * (2 ** (attempt - 1)) + random.uniform(0, 3)
            # hourly-quota wall: without a retry-after header, late attempts
            # must outwait the window, not exhaust retries inside it
            if r.status_code == 429 and attempt >= 3:
                wait = max(wait, 900.0)
            ra = r.headers.get("retry-after")
            if ra:
                try:
                    wait = max(wait, float(ra) + 1)
                except ValueError:
                    pass
            print(f"    {r.status_code}, backing off {wait:.0f}s", file=sys.stderr, flush=True)
            time.sleep(wait)
            continue
        return None, {"attempts": attempts}, f"HTTP {r.status_code}: {r.text[:400]}"
    return None, {"attempts": attempts}, "exhausted retries"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="cte_p1")
    ap.add_argument("--limit", type=int, default=None, help="cap number of calls (smoke test)")
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--tiles", default=None,
                    help="comma-separated image names; restricts the tile set")
    ap.add_argument("--reps", type=int, default=None,
                    help="replicates per tile for EVERY selected tile "
                         "(replaces the default 20+repeat-triple plan)")
    ap.add_argument("--tag", default="",
                    help="suffix for the output file, e.g. _stability")
    ap.add_argument("--full-test", action="store_true",
                    help="run every test-partition tile from coverage_index.csv "
                         "(grids rendered in memory; 1 replicate per tile)")
    args = ap.parse_args()

    api_key = os.environ.get(KEY_ENV)
    if not api_key:
        sys.exit(f"{KEY_ENV} is not set")

    spec = yaml.safe_load(open(SPEC))
    prompt_text = open(os.path.join(RENDERED, f"{args.prompt}.txt")).read()
    if args.full_test:
        tiles = load_full_test()
    else:
        manifest = json.load(open(MANIFEST))
        tiles = manifest["tiles"]
    if args.tiles:
        want = {n.strip() for n in args.tiles.split(",")}
        tiles = [t for t in tiles if t["image"] in want]
        missing = want - {t["image"] for t in tiles}
        if missing:
            sys.exit(f"unknown tiles: {sorted(missing)}")

    if args.reps:
        jobs = [(t, rep) for t in tiles for rep in range(1, args.reps + 1)]
    else:
        # main pass over all 20, then the extra repeats for the stability check
        jobs = [(t, 1) for t in tiles]
        by_name = {t["image"]: t for t in tiles}
        for name in REPEAT_TILES:
            if name in by_name:
                for rep in range(2, N_REPEATS + 1):
                    jobs.append((by_name[name], rep))
    if args.limit:
        jobs = jobs[: args.limit]

    os.makedirs(RUNS, exist_ok=True)
    out_path = os.path.join(RUNS, f"{args.prompt}{args.tag}.jsonl")
    def scan():
        done, tries = set(), {}
        if os.path.exists(out_path):
            for line in open(out_path):
                try:
                    r = json.loads(line)
                    key = (r["image"], r["replicate"])
                except (json.JSONDecodeError, KeyError):
                    continue
                # same predicate as the completeness check: a valid label, no error
                if not r.get("error") and (r.get("parsed") or {}).get("label") in ("HP", "SSA"):
                    done.add(key)
                elif counts_as_try(r.get("error")):
                    tries[key] = tries.get(key, 0) + 1
        return done, tries

    done, tries = scan()
    if done or tries:
        print(f"resuming: {len(done)} valid calls recorded, {sum(1 for v in tries.values() if v >= MAX_TRIES)} given up")

    session = requests.Session()
    last = 0.0
    n = 0
    for tile, rep in jobs:
        if (tile["image"], rep) in done or tries.get((tile["image"], rep), 0) >= MAX_TRIES:
            continue
        n += 1
        gap = MIN_INTERVAL_S - (time.time() - last)
        if gap > 0 and last:
            time.sleep(gap)
        last = time.time()

        print(f"[{n}] {tile['image']} rep{rep}", file=sys.stderr, flush=True)
        raw, meta, err = call(
            session, prompt_text, b64_gridded_tile(tile), api_key,
            temperature=args.temperature,
        )

        rec = {
            "prompt_id": args.prompt,
            "model": MODEL,
            "base_url": BASE_URL,
            "temperature": args.temperature,
            "sampling": SAMPLING or "host default",
            "extra_body": EXTRA_BODY or None,
            "image": tile["image"],
            "replicate": rep,
            "label_true": tile["label"],
            "ssa_votes_out_of_7": tile["ssa_votes_out_of_7"],
            "agreement_band": tile["agreement_band"],
            "n_empty_cells": tile["n_empty_cells"],
            "empty_cells": tile["empty_cells"],
            "raw_response": raw,
            "error": err,
            "meta": meta,
        }
        if raw is not None:
            payload, perr = extract_json(raw)
            rec["json_extract_error"] = perr
            rec["raw_key_order"] = top_level_key_order(payload) if payload else None
            if payload:
                try:
                    obj = json.loads(payload)
                    viol, norm = validate(obj, spec)
                    rec["parsed"] = norm
                    rec["violations"] = viol
                    rec["parsed_key_order"] = list(obj) if isinstance(obj, dict) else None
                except json.JSONDecodeError as e:
                    rec["parsed"] = None
                    rec["violations"] = [f"json decode failed: {e}"]
            mark_invalid(rec)

        with open(out_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    print(f"done: {n} new calls -> {out_path}")
    ok, tries = scan()
    missing = [(t["image"], rep) for t, rep in jobs if (t["image"], rep) not in ok]
    gave_up = [k for k in missing if tries.get(k, 0) >= MAX_TRIES]
    unfinished = [k for k in missing if k not in set(gave_up)]
    if gave_up:
        print(f"GAVE UP on {len(gave_up)} calls after {MAX_TRIES} unusable responses each: {gave_up[:10]}")
    if unfinished:
        print(f"INCOMPLETE: {len(unfinished)} planned calls not yet done: {unfinished[:10]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
