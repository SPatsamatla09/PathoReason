"""Score a pathoreason run for grid-localisation ability.

Answers five questions, each against an explicit null rather than an eyeballed
threshold:

  1. cell-name validity          -- are cited cells inside A1-D4 at all
  2. cross-image variation       -- do citations track content or repeat regardless
  3. within-tile stability       -- do repeated runs cite the same cells
  4. empty-cell control          -- rate of citing cells with <5% tissue
  5. transposition               -- is row/column systematically swapped

The load-bearing baseline for 2 and 4 is the SHUFFLE null: re-pair each
response's cited cell set with a different tile and recompute. If the model is
localising, its real pairing beats the shuffled one. If real and shuffled agree,
the cell choices carry no image-specific information, whatever their accuracy
looks like in isolation.

    python3 score.py --run runs/cte_p1.jsonl
"""

import argparse
import itertools
import json
import os
import random
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
COVERAGE = os.path.join(ROOT, "coverage")
EMPTY_PCT = 5.0
N_PERM = 20000
SEED = 20260806

ROWS = "ABCD"
ALL_CELLS = [f"{r}{c}" for r in ROWS for c in range(1, 5)]


def transpose(cell):
    """(row, col) -> (col, row). B3 -> C2."""
    r = ROWS.index(cell[0])
    c = int(cell[1]) - 1
    return f"{ROWS[c]}{r + 1}"


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def load(run_path):
    recs = [json.loads(l) for l in open(run_path) if l.strip()]
    cov = {}
    for r in recs:
        if r["image"] not in cov:
            p = os.path.join(COVERAGE, r["image"].replace(".png", ".json"))
            cov[r["image"]] = json.load(open(p))
    return recs, cov


def cited_cells(rec):
    """Unique valid cells cited anywhere in a response."""
    p = rec.get("parsed")
    if not p:
        return []
    out = []
    for e in p.get("evidence") or []:
        out += e.get("grid_cells_valid") or []
    return sorted(set(out))


def citation_pairs(rec):
    """(feature, cell) pairs -- one response may cite a cell under several features."""
    p = rec.get("parsed")
    if not p:
        return []
    return [
        (e.get("feature"), c)
        for e in p.get("evidence") or []
        for c in (e.get("grid_cells_valid") or [])
    ]


def q1_validity(recs):
    n_resp = len(recs)
    parsed = [r for r in recs if r.get("parsed")]
    bad_cells = Counter()
    resp_with_bad = 0
    total_valid = total_invalid = 0
    for r in parsed:
        inval = [c for e in r["parsed"]["evidence"] for c in (e.get("grid_cells_invalid") or [])]
        val = [c for e in r["parsed"]["evidence"] for c in (e.get("grid_cells_valid") or [])]
        total_valid += len(val)
        total_invalid += len(inval)
        if inval:
            resp_with_bad += 1
            bad_cells.update(map(str, inval))
    viol = Counter()
    for r in recs:
        for v in r.get("violations") or []:
            viol[v.split(" ")[0].split("[")[0]] += 1
    return {
        "responses": n_resp,
        "parsed_ok": len(parsed),
        "parse_failures": n_resp - len(parsed),
        "citations_valid": total_valid,
        "citations_invalid": total_invalid,
        "invalid_rate": total_invalid / max(total_valid + total_invalid, 1),
        "responses_with_invalid_cell": resp_with_bad,
        "invalid_cell_examples": bad_cells.most_common(10),
        "violation_kinds": viol.most_common(10),
    }


def q2_variation(recs, rng):
    """Do cited cells differ by image, or is one cell set reused everywhere?"""
    first = [r for r in recs if r["replicate"] == 1 and cited_cells(r)]
    sets = {r["image"]: set(cited_cells(r)) for r in first}
    names = sorted(sets)
    freq = Counter()
    for s in sets.values():
        freq.update(s)

    pairs = list(itertools.combinations(names, 2))
    observed = sum(jaccard(sets[a], sets[b]) for a, b in pairs) / max(len(pairs), 1)

    # null: same per-response citation counts, cells drawn uniformly at random
    sizes = [len(sets[n]) for n in names]
    null = []
    for _ in range(2000):
        rand = [set(rng.sample(ALL_CELLS, k)) for k in sizes]
        null.append(
            sum(jaccard(rand[i], rand[j]) for i, j in itertools.combinations(range(len(rand)), 2))
            / max(len(pairs), 1)
        )
    null_mean = sum(null) / len(null)
    p = sum(1 for v in null if v >= observed) / len(null)

    return {
        "n_images": len(names),
        "mean_pairwise_jaccard_between_images": round(observed, 4),
        "null_uniform_random_jaccard": round(null_mean, 4),
        "p_observed_ge_null": round(p, 4),
        "cells_never_cited": [c for c in ALL_CELLS if freq[c] == 0],
        "cell_frequency": {c: freq[c] for c in ALL_CELLS},
        "mean_cells_cited_per_response": round(sum(sizes) / max(len(sizes), 1), 2),
    }


def q3_stability(recs):
    by = defaultdict(list)
    for r in recs:
        if cited_cells(r):
            by[r["image"]].append(r)
    out = {}
    within = []
    for img, rs in sorted(by.items()):
        if len(rs) < 2:
            continue
        rs.sort(key=lambda r: r["replicate"])
        sets = [set(cited_cells(r)) for r in rs]
        js = [jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
        within += js
        out[img] = {
            "n_reps": len(rs),
            "labels": [r["parsed"]["label"] for r in rs],
            "confidences": [r["parsed"]["confidence"] for r in rs],
            "cited_per_rep": [sorted(s) for s in sets],
            "mean_pairwise_jaccard": round(sum(js) / len(js), 4),
            "cells_in_every_rep": sorted(set.intersection(*sets)),
            "cells_in_only_one_rep": sorted(
                c for c in set.union(*sets) if sum(c in s for s in sets) == 1
            ),
        }
    return {
        "per_image": out,
        "mean_within_image_jaccard": round(sum(within) / len(within), 4) if within else None,
    }


def q4_empty_control(recs, cov, rng):
    """Rate of citing cells that contain essentially no tissue."""
    rows = []
    for r in recs:
        cells = cited_cells(r)
        if not cells:
            continue
        c = cov[r["image"]]
        empty = set(c["empty_cells"])
        hits = [x for x in cells if x in empty]
        rows.append(
            {
                "image": r["image"],
                "replicate": r["replicate"],
                "cells": cells,
                "n_cited": len(cells),
                "n_empty_available": len(empty),
                "n_cited_empty": len(hits),
                "cited_empty": hits,
                "min_tissue_pct_cited": min(c["cells"][x]["tissue_pct"] for x in cells),
                "mean_tissue_pct_cited": round(
                    sum(c["cells"][x]["tissue_pct"] for x in cells) / len(cells), 2
                ),
            }
        )

    tot_cited = sum(x["n_cited"] for x in rows)
    tot_empty_hits = sum(x["n_cited_empty"] for x in rows)

    # null A: cells drawn uniformly at random from the 16 in the SAME image
    uni = []
    for _ in range(N_PERM // 4):
        s = 0
        for x in rows:
            emp = set(cov[x["image"]]["empty_cells"])
            s += sum(1 for c in rng.sample(ALL_CELLS, x["n_cited"]) if c in emp)
        uni.append(s)
    uni_mean = sum(uni) / len(uni)
    p_uni = sum(1 for v in uni if v <= tot_empty_hits) / len(uni)

    # null B (stronger): keep each response's cell set, re-pair it with a
    # different tile. Isolates image-specific information from prior cell habits.
    imgs = [x["image"] for x in rows]
    shuf = []
    for _ in range(N_PERM // 4):
        perm = imgs[:]
        rng.shuffle(perm)
        s = 0
        for x, other in zip(rows, perm):
            emp = set(cov[other]["empty_cells"])
            s += sum(1 for c in x["cells"] if c in emp)
        shuf.append(s)
    shuf_mean = sum(shuf) / len(shuf)
    p_shuf = sum(1 for v in shuf if v <= tot_empty_hits) / len(shuf)

    per_pair = Counter()
    for r in recs:
        c = cov[r["image"]]
        for feat, cell in citation_pairs(r):
            if c["cells"][cell]["tissue_pct"] < EMPTY_PCT:
                per_pair[feat] += 1

    return {
        "responses_scored": len(rows),
        "total_cells_cited": tot_cited,
        "cells_cited_that_are_empty": tot_empty_hits,
        "empty_citation_rate": round(tot_empty_hits / max(tot_cited, 1), 4),
        "null_uniform_random_expected": round(uni_mean, 2),
        "null_uniform_rate": round(uni_mean / max(tot_cited, 1), 4),
        "p_observed_le_uniform_null": round(p_uni, 4),
        "null_shuffled_image_expected": round(shuf_mean, 2),
        "null_shuffled_rate": round(shuf_mean / max(tot_cited, 1), 4),
        "p_observed_le_shuffled_null": round(p_shuf, 4),
        "empty_citations_by_feature": per_pair.most_common(),
        "per_response": rows,
    }


def q5_transposition(recs, cov):
    """If the model swaps row and column, the transposed cell should fit better."""
    same = both = as_is_better = transposed_better = 0
    tissue_as_is = tissue_transposed = 0.0
    n = 0
    empty_hits_rescued = 0
    empty_hits_total = 0
    for r in recs:
        c = cov[r["image"]]
        for cell in cited_cells(r):
            t = transpose(cell)
            n += 1
            a = c["cells"][cell]["tissue_pct"]
            b = c["cells"][t]["tissue_pct"]
            tissue_as_is += a
            tissue_transposed += b
            if cell == t:
                same += 1
                continue
            both += 1
            if a > b:
                as_is_better += 1
            elif b > a:
                transposed_better += 1
            if a < EMPTY_PCT:
                empty_hits_total += 1
                if b >= EMPTY_PCT:
                    empty_hits_rescued += 1
    return {
        "citations_examined": n,
        "on_diagonal_transpose_is_identity": same,
        "off_diagonal": both,
        "as_is_has_more_tissue": as_is_better,
        "transposed_has_more_tissue": transposed_better,
        "mean_tissue_pct_as_cited": round(tissue_as_is / max(n, 1), 2),
        "mean_tissue_pct_if_transposed": round(tissue_transposed / max(n, 1), 2),
        "empty_citations_off_diagonal": empty_hits_total,
        "empty_citations_whose_transpose_is_non_empty": empty_hits_rescued,
        "rescue_rate": round(empty_hits_rescued / max(empty_hits_total, 1), 4),
    }


def classification(recs):
    first = [r for r in recs if r["replicate"] == 1 and r.get("parsed")]
    ok = [r for r in first if r["parsed"]["label"] in ("HP", "SSA")]
    corr = [r for r in ok if r["parsed"]["label"] == r["label_true"]]
    by_band = defaultdict(lambda: [0, 0])
    pred = Counter()
    for r in ok:
        pred[r["parsed"]["label"]] += 1
        b = by_band[r["agreement_band"]]
        b[1] += 1
        if r["parsed"]["label"] == r["label_true"]:
            b[0] += 1
    return {
        "n": len(ok),
        "accuracy": round(len(corr) / max(len(ok), 1), 4),
        "predicted_distribution": dict(pred),
        "true_distribution": dict(Counter(r["label_true"] for r in ok)),
        "accuracy_by_band": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_band.items())},
        "mean_confidence": round(
            sum(float(r["parsed"]["confidence"] or 0) for r in ok) / max(len(ok), 1), 3
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/cte_p1.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = random.Random(SEED)
    recs, cov = load(os.path.join(ROOT, args.run))

    report = {
        "run": args.run,
        "model": recs[0]["model"] if recs else None,
        "prompt_id": recs[0]["prompt_id"] if recs else None,
        "temperature": recs[0].get("temperature") if recs else None,
        "q1_cell_validity": q1_validity(recs),
        "q2_cross_image_variation": q2_variation(recs, rng),
        "q3_within_tile_stability": q3_stability(recs),
        "q4_empty_cell_control": q4_empty_control(recs, cov, rng),
        "q5_transposition": q5_transposition(recs, cov),
        "classification": classification(recs),
    }
    out = os.path.join(ROOT, args.out or args.run.replace(".jsonl", "_score.json"))
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "q4_empty_cell_control"}, indent=2)[:200])
    print(f"\nfull report -> {out}")
    return report


if __name__ == "__main__":
    main()
