"""Score gemma's cte_p1 serration and crypt-architecture descriptions
against rubric.yaml.

The rubric makes both features primary ONLY when the description carries
direction: serration must be located on the crypt axis (base-extending -> SSA,
upper/surface-confined -> HP, unlocated -> neither); architecture must assert
or deny the basal-distortion criteria (dilation, horizontal, L/boot shapes).

Location / direction classes are keyword rules over the exact strings in
runs/cte_p1.jsonl, listed in the output so every assignment is auditable.

    python3 score_descriptions.py   ->  runs/description_scoring.json
"""

import json
import os
import re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(ROOT, "runs", "cte_p1.jsonl")
OUT = os.path.join(ROOT, "runs", "description_scoring.json")

UPPER = ("upper", "superficial", "near the surface")
DEEP = ("deep into", "deeper parts", "extends deep", "to the base", "along the base")
THROUGHOUT = ("throughout the crypts", "throughout the upper and middle")

SSA_ARCH = (
    "basal dilat", "basal expansion", "boot-shaped", "l-shaped", "horizontal",
    "broadened base", "flattened, broadened", "distorted",
)
HP_ARCH = ("straight", "parallel", "without the basal", "no basal", "lacking the basal",
           "do not show basal", "not show basal")


def classify_serration(desc):
    d = desc.lower()
    if any(k in d for k in UPPER):
        return "upper_confined", "HP"
    if any(k in d for k in DEEP):
        return "base_extending", "SSA"
    if "throughout the crypts" in d or "throughout these" in d and False:
        return "throughout_implicit_base", "SSA"
    if "throughout the crypts" in d:
        return "throughout_implicit_base", "SSA"
    return "no_location", None


def classify_arch(desc):
    d = desc.lower()
    ssa_hit = any(k in d for k in SSA_ARCH)
    hp_hit = any(k in d for k in HP_ARCH)
    # negated-SSA phrasing ("without the basal dilation seen in SSA") names the
    # criterion to DENY it -> HP direction, so HP keywords win when both match
    if hp_hit:
        return "orderly_narrow_bases", "HP"
    if ssa_hit:
        return "basal_distortion", "SSA"
    return "generic", None


def criterion_terms(desc):
    d = desc.lower()
    return [
        t for t, pat in [
            ("basal dilation", r"basal (dilat|expansion)"),
            ("boot-shaped", r"boot-shaped"),
            ("L-shaped", r"l-shaped"),
            ("horizontal growth", r"horizontal"),
            ("branching", r"branch"),
        ] if re.search(pat, d)
    ]


def main():
    recs = [json.loads(l) for l in open(RUN)]
    items = defaultdict(list)
    for r in recs:
        for e in (r.get("parsed") or {}).get("evidence", []):
            if e["feature"] in ("serration", "crypt/gland architecture"):
                items[e["feature"]].append(
                    {
                        "tile": r["image"][6:9],
                        "replicate": r["replicate"],
                        "pred": r["parsed"]["label"],
                        "true": r["label_true"],
                        "desc": e["description"],
                    }
                )

    report = {}
    for feat, cls in [("serration", classify_serration),
                      ("crypt/gland architecture", classify_arch)]:
        rows = []
        for it in items[feat]:
            c, direction = cls(it["desc"])
            rows.append(
                {
                    **it,
                    "class": c,
                    "direction": direction,
                    "rubric_relevance_as_described": "primary" if direction else "none",
                    "direction_matches_pred": direction == it["pred"] if direction else None,
                    "direction_matches_truth": direction == it["true"] if direction else None,
                    "criterion_terms": criterion_terms(it["desc"]) if "crypt" in feat else [],
                }
            )
        directional = [r for r in rows if r["direction"]]
        report[feat] = {
            "n_items": len(rows),
            "class_counts": dict(Counter(r["class"] for r in rows)),
            "directional": len(directional),
            "undirected_downgraded_to_none": len(rows) - len(directional),
            "direction_matches_stated_label": f"{sum(r['direction_matches_pred'] for r in directional)}/{len(directional)}",
            "direction_matches_true_label": f"{sum(r['direction_matches_truth'] for r in directional)}/{len(directional)}",
            "inconsistent_items": [
                {k: r[k] for k in ("tile", "replicate", "pred", "class", "desc")}
                for r in directional if not r["direction_matches_pred"]
            ],
            "items": rows,
        }

    arch_terms = Counter(t for r in report["crypt/gland architecture"]["items"]
                         for t in r["criterion_terms"])
    report["crypt/gland architecture"]["criterion_term_usage"] = dict(arch_terms)

    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
    for feat in report:
        d = {k: v for k, v in report[feat].items() if k != "items"}
        print(feat.upper());
        print(json.dumps(d, indent=1)[:1200]);
        print()
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
