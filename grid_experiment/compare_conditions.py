"""Compare cited evidence across ordering conditions on the same tiles.

The question is whether classify-then-explain and explain-then-classify produce
different evidence, or the same evidence with the keys reordered. A raw
similarity number cannot answer that on its own, so every cross-condition
figure is bracketed by two references measured from the same data:

  floor   between-image similarity within one condition
          -- what unrelated evidence looks like
  ceiling within-image, within-condition similarity across replicates
          -- what pure sampling noise looks like at temperature 1.0

Cross-condition similarity at the ceiling means the manipulation moved nothing
but key order. At the floor it means the orderings produced unrelated evidence.

    python3 compare_conditions.py --a runs/cte_p1.jsonl --b runs/etc_p1.jsonl
"""

import argparse
import itertools
import json
import os
import random
import re
from collections import Counter, defaultdict

from score import cited_cells, jaccard, load

ROOT = os.path.dirname(os.path.abspath(__file__))
SEED = 20260806
STOP = set(
    "the a an of in on at to and or is are with this that these those it its as "
    "show shows showing seen visible present within throughout across some few "
    "there their has have been be by from for which where while".split()
)


def features(rec):
    p = rec.get("parsed") or {}
    return [e.get("feature") for e in (p.get("evidence") or []) if e.get("feature")]


def descriptions(rec):
    p = rec.get("parsed") or {}
    return [e.get("description") or "" for e in (p.get("evidence") or [])]


def content_tokens(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP and len(w) > 2}


def by_image(recs, replicate=1):
    return {r["image"]: r for r in recs if r["replicate"] == replicate}


def within_condition_ceiling(recs):
    """Replicate-to-replicate agreement inside one condition."""
    by = defaultdict(list)
    for r in recs:
        if cited_cells(r):
            by[r["image"]].append(r)
    cells, feats = [], []
    for rs in by.values():
        if len(rs) < 2:
            continue
        for a, b in itertools.combinations(rs, 2):
            cells.append(jaccard(cited_cells(a), cited_cells(b)))
            feats.append(jaccard(features(a), features(b)))
    return (
        sum(cells) / len(cells) if cells else None,
        sum(feats) / len(feats) if feats else None,
        len(cells),
    )


def between_image_floor(recs):
    firsts = by_image(recs)
    names = [n for n in sorted(firsts) if cited_cells(firsts[n])]
    cells, feats = [], []
    for a, b in itertools.combinations(names, 2):
        cells.append(jaccard(cited_cells(firsts[a]), cited_cells(firsts[b])))
        feats.append(jaccard(features(firsts[a]), features(firsts[b])))
    return (
        sum(cells) / len(cells) if cells else None,
        sum(feats) / len(feats) if feats else None,
        len(cells),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="runs/cte_p1.jsonl")
    ap.add_argument("--b", default="runs/etc_p1.jsonl")
    ap.add_argument("--co", default="runs/co_p1.jsonl")
    args = ap.parse_args()
    rng = random.Random(SEED)

    A, cov = load(os.path.join(ROOT, args.a))
    B, _ = load(os.path.join(ROOT, args.b))
    a1, b1 = by_image(A), by_image(B)
    shared = sorted(set(a1) & set(b1))

    # the manipulation must actually have been delivered
    ko_a = Counter(tuple(r.get("raw_key_order") or []) for r in A)
    ko_b = Counter(tuple(r.get("raw_key_order") or []) for r in B)

    per_tile = []
    for img in shared:
        ra, rb = a1[img], b1[img]
        ca, cb = cited_cells(ra), cited_cells(rb)
        fa, fb = features(ra), features(rb)
        ta = content_tokens(" ".join(descriptions(ra)))
        tb = content_tokens(" ".join(descriptions(rb)))
        pa = ra.get("parsed") or {}
        pb = rb.get("parsed") or {}
        per_tile.append(
            {
                "image": img,
                "label_true": ra["label_true"],
                "label_a": pa.get("label"),
                "label_b": pb.get("label"),
                "labels_agree": pa.get("label") == pb.get("label"),
                "conf_a": pa.get("confidence"),
                "conf_b": pb.get("confidence"),
                "n_evidence_a": len(fa),
                "n_evidence_b": len(fb),
                "n_cells_a": len(ca),
                "n_cells_b": len(cb),
                "cells_a": ca,
                "cells_b": cb,
                "cell_jaccard": round(jaccard(ca, cb), 4),
                "feature_jaccard": round(jaccard(fa, fb), 4),
                "description_token_jaccard": round(jaccard(ta, tb), 4),
            }
        )

    def mean(k):
        v = [t[k] for t in per_tile if t[k] is not None]
        return round(sum(v) / len(v), 4) if v else None

    ceil_a = within_condition_ceiling(A)
    ceil_b = within_condition_ceiling(B)
    floor_a = between_image_floor(A)
    floor_b = between_image_floor(B)

    cross_cell = mean("cell_jaccard")
    ceiling = [x for x in (ceil_a[0], ceil_b[0]) if x is not None]
    ceiling = sum(ceiling) / len(ceiling) if ceiling else None
    floor = [x for x in (floor_a[0], floor_b[0]) if x is not None]
    floor = sum(floor) / len(floor) if floor else None

    position = None
    if ceiling is not None and floor is not None and ceiling > floor:
        position = round((cross_cell - floor) / (ceiling - floor), 4)

    # is cross-condition agreement lower than within-condition replicate noise?
    within_vals = []
    for recs in (A, B):
        by = defaultdict(list)
        for r in recs:
            if cited_cells(r):
                by[r["image"]].append(r)
        for rs in by.values():
            for x, y in itertools.combinations(rs, 2):
                within_vals.append(jaccard(cited_cells(x), cited_cells(y)))
    cross_vals = [t["cell_jaccard"] for t in per_tile]
    obs_diff = (sum(within_vals) / len(within_vals)) - (sum(cross_vals) / len(cross_vals))
    pool = within_vals + cross_vals
    hits = 0
    for _ in range(20000):
        rng.shuffle(pool)
        x = pool[: len(within_vals)]
        y = pool[len(within_vals) :]
        if (sum(x) / len(x)) - (sum(y) / len(y)) >= obs_diff:
            hits += 1

    feat_a = Counter(f for r in A if r["replicate"] == 1 for f in features(r))
    feat_b = Counter(f for r in B if r["replicate"] == 1 for f in features(r))

    def acc(recs):
        f = [r for r in recs if r["replicate"] == 1 and (r.get("parsed") or {}).get("label")]
        n = len(f)
        k = sum(1 for r in f if r["parsed"]["label"] == r["label_true"])
        conf = [float(r["parsed"]["confidence"] or 0) for r in f]
        return {
            "n": n,
            "correct": k,
            "accuracy": round(k / max(n, 1), 4),
            "mean_confidence": round(sum(conf) / max(len(conf), 1), 3),
            "predicted": dict(Counter(r["parsed"]["label"] for r in f)),
        }

    report = {
        "condition_a": os.path.basename(args.a),
        "condition_b": os.path.basename(args.b),
        "manipulation_delivered": {
            "a_raw_key_orders": {"|".join(k): v for k, v in ko_a.items()},
            "b_raw_key_orders": {"|".join(k): v for k, v in ko_b.items()},
        },
        "cited_cells": {
            "cross_condition_jaccard": cross_cell,
            "ceiling_within_condition_replicates": round(ceiling, 4) if ceiling else None,
            "floor_between_images": round(floor, 4) if floor else None,
            "position_floor0_ceiling1": position,
            "permutation_p_within_gt_cross": round(hits / 20000, 4),
            "n_within_pairs": len(within_vals),
            "n_cross_pairs": len(cross_vals),
        },
        "features": {
            "cross_condition_jaccard": mean("feature_jaccard"),
            "ceiling_within_condition": round(ceil_a[1], 4) if ceil_a[1] else None,
            "floor_between_images": round(floor_a[1], 4) if floor_a[1] else None,
            "usage_a": feat_a.most_common(),
            "usage_b": feat_b.most_common(),
        },
        "descriptions": {"cross_condition_token_jaccard": mean("description_token_jaccard")},
        "verbosity": {
            "mean_evidence_items_a": mean("n_evidence_a"),
            "mean_evidence_items_b": mean("n_evidence_b"),
            "mean_cells_cited_a": mean("n_cells_a"),
            "mean_cells_cited_b": mean("n_cells_b"),
        },
        "labels": {
            "agreement_rate": round(
                sum(1 for t in per_tile if t["labels_agree"]) / max(len(per_tile), 1), 4
            ),
            "accuracy_a": acc(A),
            "accuracy_b": acc(B),
        },
        "per_tile": per_tile,
    }

    co_path = os.path.join(ROOT, args.co)
    if os.path.exists(co_path):
        C, _ = load(co_path)
        report["labels"]["accuracy_classify_only"] = acc(C)
        report["manipulation_delivered"]["co_raw_key_orders"] = {
            "|".join(k): v for k, v in Counter(tuple(r.get("raw_key_order") or []) for r in C).items()
        }

    out = os.path.join(ROOT, "runs", "condition_comparison.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "per_tile"}, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
