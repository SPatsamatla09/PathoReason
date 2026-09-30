"""Check that a new endpoint can stand in for the archived Cerebras gemma-4-31b.

Uses the same PATHO_BASE_URL / PATHO_MODEL / PATHO_KEY_ENV / PATHO_MAX_TOKENS_PARAM
settings as the runners and runs four checks, spending 4-5 calls:

  1. text call works (and which max-tokens parameter name the host accepts)
  2. vision works: the model reads the grid-line colour off a gridded tile
  3. the real classify-then-explain prompt returns parseable, schema-valid JSON
     with the requested key order and valid grid cells
  4. rate-limit headers, if the host sends any

Exit code 0 only if 1-3 pass.

    PATHO_BASE_URL=https://openrouter.ai/api/v1 PATHO_MODEL=google/gemma-4-31b-it \\
    PATHO_KEY_ENV=OPENROUTER_API_KEY python3 probe_host.py
"""

import json
import os
import sys

import requests
import yaml

import run_experiment as rx

ROOT = os.path.dirname(os.path.abspath(__file__))


def post(body, key):
    return requests.post(f"{rx.BASE_URL}/chat/completions", json=body, timeout=180,
                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})


def main():
    key = os.environ.get(rx.KEY_ENV)
    if not key:
        sys.exit(f"FAIL  {rx.KEY_ENV} is not set")
    print(f"host  {rx.BASE_URL}  model {rx.MODEL}  pinned provider {rx.PROVIDER}  -> host_tag {rx.host_tag()}")
    ok = True
    if "openrouter" in rx.BASE_URL and not rx.PROVIDER:
        sys.exit("FAIL  OpenRouter can route one model id to several upstreams; set PATHO_PROVIDER "
                 "(e.g. the provider slug from openrouter.ai/<model>) so every call hits the same one")

    param = rx.MAX_TOKENS_PARAM
    pin0 = {"provider": {"order": [rx.PROVIDER], "allow_fallbacks": False}} if rx.PROVIDER else {}
    r = post({"model": rx.MODEL, param: 20, **pin0, "messages": [{"role": "user", "content": "Reply with the word OK."}]}, key)
    if r.status_code == 400 and param in r.text:
        alt = "max_tokens" if param != "max_tokens" else "max_completion_tokens"
        r2 = post({"model": rx.MODEL, alt: 20, **pin0, "messages": [{"role": "user", "content": "Reply with the word OK."}]}, key)
        if r2.status_code == 200:
            print(f"NOTE  host rejects '{param}', accepts '{alt}': export PATHO_MAX_TOKENS_PARAM={alt}")
            param, r = alt, r2
    try:
        d = r.json() if r.status_code == 200 else None
        d["choices"][0]["message"]
    except (ValueError, KeyError, IndexError, TypeError):
        print(f"FAIL  text call HTTP {r.status_code}: {r.text[:300]}")
        sys.exit(1)
    served = d.get("provider")
    print(f"PASS  text call; served model id = {d.get('model')!r}; served by provider = {served!r}")
    if rx.PROVIDER and served and rx.PROVIDER.split("/")[0].lower() not in str(served).lower():
        print(f"FAIL  pinned provider {rx.PROVIDER!r} but served by {served!r}")
        ok = False
    limits = {k: v for k, v in r.headers.items() if "ratelimit" in k.lower() or k.lower() == "retry-after"}

    img = rx.b64_png(os.path.join(ROOT, "gridded", "MHIST_ccn_grid.png"))
    pin = {"provider": {"order": [rx.PROVIDER], "allow_fallbacks": False}} if rx.PROVIDER else {}
    r = post({"model": rx.MODEL, param: 20, "temperature": 0, **pin, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
        {"type": "text", "text": "What colour are the thin grid lines drawn on this image? Answer with one word."}]}]}, key)
    try:
        ans = r.json()["choices"][0]["message"].get("content") or ""
    except (ValueError, KeyError, IndexError, TypeError):
        ans = r.text[:200]
    if r.status_code == 200 and "green" in ans.lower():
        print(f"PASS  vision: {ans.strip()!r}")
    else:
        print(f"FAIL  vision: HTTP {r.status_code} {ans!r}")
        ok = False

    spec = yaml.safe_load(open(rx.SPEC))
    prompt = open(os.path.join(rx.RENDERED, "cte_p1.txt")).read()
    # temperature-1.0 JSON compliance is stochastic: allow up to 3 tries
    for attempt in range(3):
        raw, meta, err = rx.call(requests.Session(), prompt, img, key)
        payload = rx.extract_json(raw)[0] if raw else None
        if not err and payload:
            break
    if err:
        print(f"FAIL  cte_p1 call: {err}")
        ok = False
    else:
        payload, perr = rx.extract_json(raw)
        if not payload:
            print(f"FAIL  cte_p1 JSON not extractable ({perr}): {raw[:200]!r}")
            ok = False
        else:
            viol, parsed = rx.validate(json.loads(payload), spec)
            order = rx.top_level_key_order(payload)
            status = "PASS" if not viol and order[:1] == ["label"] else "FAIL"
            ok &= status == "PASS"
            cells = sorted({c for e in parsed["evidence"] for c in e["grid_cells_valid"]})
            print(f"{status}  cte_p1: label={parsed['label']} conf={parsed['confidence']} "
                  f"key order={order} violations={viol or 'none'} cells={cells}")
            print(f"      usage={meta.get('usage')}")

    print(f"info  rate-limit headers: {limits or 'none sent'}")
    print("RESULT", "OK" if ok else "NOT USABLE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
