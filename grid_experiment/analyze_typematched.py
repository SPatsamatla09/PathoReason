"""Tissue-type-matched control arm vs the original area-matched arm (Friendli extension tiles).

Tile-level exact sign test, as in every other masking analysis:
    net = sum over occlusions of (cited flip - control flip); flip = label != baseline label
Reported:
  1. cited vs TYPE-matched: the new run, both arms interleaved in one run, 262 usable tiles
  2. cited vs AREA-matched: the original extension run, restricted to the same 262 tiles
  3. test-retest of the cited arm: identical cited images, run twice (original vs new run)
  4. the type-matched result where the control holds no cell the model cited anywhere,
     vs where it holds 1 or more
  5. how far the area-matched controls were from the cited cells in epithelium

    python3 analyze_typematched.py  ->  runs/masking_typematched_analysis.json
"""

import glob
import json
import math
import os
import statistics

ROOT = os.path.dirname(os.path.abspath(__file__))
H = "openrouter__google-gemma-4-31b-it__friendli"
OCCS = ("mean", "blur", "black")


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def ok(r):
    return (not r.get("error") and (r.get("parsed") or {}).get("label") in ("HP", "SSA")
            and (r.get("baseline") or {}).get("label") in ("HP", "SSA"))


def sign_p(a, b):
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def test(cell, tiles, ctrl):
    pos = neg = tie = cf = kf = n = 0
    used = []
    for t in tiles:
        pr = []
        for o in OCCS:
            a, b = cell.get((t, "cited", o)), cell.get((t, ctrl, o))
            if not a or not b:
                break
            pr.append((a["parsed"]["label"] != a["baseline"]["label"], b["parsed"]["label"] != b["baseline"]["label"]))
        if len(pr) != 3:
            continue
        used.append(t)
        net = sum(int(x) - int(y) for x, y in pr)
        pos, neg, tie = pos + (net > 0), neg + (net < 0), tie + (net == 0)
        cf, kf, n = cf + sum(x for x, _ in pr), kf + sum(y for _, y in pr), n + 3
    return {"n_tiles": len(used), "cited_more": pos, "control_more": neg, "tied": tie,
            "sign_test_p": float(f"{sign_p(pos, neg):.3g}"),
            "cited_flip_rate": round(cf / max(n, 1), 4), "control_flip_rate": round(kf / max(n, 1), 4)}


def main():
    new = {(r["tile_dir"], r["arm"], r["occlusion"]): r
           for r in rows(os.path.join(ROOT, "runs", f"masking_typematched__{H}_k3.jsonl")) if ok(r)}
    old = {(r["tile_dir"], r["arm"], r["occlusion"]): r
           for r in rows(os.path.join(ROOT, "runs", f"masking_ext__{H}_k3.jsonl")) if ok(r)}
    mdir = os.path.join(ROOT, "masked", f"typematched__{H}_k3")
    man = {os.path.basename(os.path.dirname(f)): json.load(open(f)) for f in glob.glob(mdir + "/*/manifest.json")}
    tiles = sorted(man)
    rep = {"n_usable_tiles": len(tiles),
           "n_unusable_tiles": len(glob.glob(mdir + "_unusable/*/manifest.json")),
           "cited_vs_type_matched": test(new, tiles, "type_matched"),
           "cited_vs_area_matched_same_tiles_original_run": test(old, tiles, "tissue_matched")}
    # cited-arm test-retest: same images, two runs
    agree = tot = 0
    for t in tiles:
        for o in OCCS:
            a, b = new.get((t, "cited", o)), old.get((t, "cited", o))
            if a and b:
                tot += 1
                agree += a["parsed"]["label"] == b["parsed"]["label"]
    rep["cited_arm_test_retest_label_agreement"] = f"{agree}/{tot}" + (f" ({agree / tot:.3f})" if tot else "")
    # control made only of cells the model never cited anywhere
    clean = [t for t in tiles if man[t]["type_matched"]["cells_in_full_citation"] == 0]
    rep["type_matched_fully_uncited_controls"] = test(new, clean, "type_matched")
    rep["type_matched_controls_with_cited_cells"] = test(new, [t for t in tiles if t not in clean], "type_matched")
    # tissue-type mismatch of the old controls, all tiles in the source sweep
    allm = man.copy()
    for f in glob.glob(mdir + "_unusable/*/manifest.json"):
        allm[os.path.basename(os.path.dirname(f))] = json.load(open(f))
    gaps = sorted(m["area_matched_control_for_reference"]["epithelium_gap_pp"] for m in allm.values())
    rep["area_matched_epithelium_gap_pp"] = {
        "median": round(statistics.median(gaps), 3), "control_less_epithelial": sum(g < 0 for g in gaps),
        "abs_gap_over_5pp": sum(abs(g) > 5 for g in gaps), "n": len(gaps)}
    tm = sorted(man[t]["type_matched"]["epithelium_gap_pp"] for t in tiles)
    rep["type_matched_epithelium_gap_pp"] = {"median": round(statistics.median(tm), 3), "max_abs": max(abs(g) for g in tm)}
    out = os.path.join(ROOT, "runs", "masking_typematched_analysis.json")
    json.dump(rep, open(out, "w"), indent=2)
    print(json.dumps(rep, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
