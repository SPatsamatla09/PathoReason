"""Unblind and score rater CSVs from rating_form.html against the key.

    python3 score_ratings.py rating/ratings_RR.csv [rating/ratings_XX.csv ...]

Per rater: completeness, then each axis broken out by model (gemma vs gpt-4o),
by the feature cited, and by whether the model's stated label matched the
consensus. With two raters, adds per-axis agreement and Cohen's kappa.
The pre-registered prediction: relevance and support near ceiling, the
unfaithfulness surfacing on presence.
"""

import csv
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
KEY = os.path.join(ROOT, "rating", "KEY_do_not_open_until_scored.csv")
SHEET = os.path.join(ROOT, "rating", "rater_A.csv")  # item text, identical to rater_B
AXES = ["presence", "relevance", "support"]


def kappa(a, b):
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def load(path):
    rows = {r["rating_id"]: r for r in csv.DictReader(open(path))}
    return rows


def main():
    key = {r["rating_id"]: r for r in csv.DictReader(open(KEY))}
    sheet = {r["rating_id"]: r for r in csv.DictReader(open(SHEET))}
    raters = {os.path.basename(p): load(os.path.join(ROOT, p)) for p in sys.argv[1:]}
    if not raters:
        sys.exit("give at least one rater CSV")

    for name, rows in raters.items():
        done = [i for i in key if all(rows.get(i, {}).get(ax) for ax in AXES)]
        print(f"\n=== {name}: {len(done)}/{len(key)} items fully scored ===")
        if len(done) < len(key):
            print("  incomplete ids:", [i for i in key if i not in done][:10], "...")
        by_model = defaultdict(lambda: defaultdict(Counter))
        by_feat = defaultdict(lambda: defaultdict(Counter))
        by_correct = defaultdict(lambda: defaultdict(Counter))
        for i in done:
            k, s, r = key[i], sheet[i], rows[i]
            model = "gpt4o" if k["run"].startswith("gpt4o") else "gemma"
            correct = "model_correct" if s["stated_label"] == k["true_label"] else "model_wrong"
            for ax in AXES:
                by_model[ax][model][r[ax]] += 1
                by_feat[ax][s["feature"]][r[ax]] += 1
                by_correct[ax][correct][r[ax]] += 1
        for ax in AXES:
            print(f"\n  {ax.upper()}")
            for grp, title in ((by_model, "by model"), (by_correct, "by label correctness"), (by_feat, "by cited feature")):
                print(f"    {title}:")
                for g, c in sorted(grp[ax].items()):
                    tot = sum(c.values())
                    print(f"      {g:26s} " + "  ".join(f"{v}={n} ({n/tot*100:.0f}%)" for v, n in sorted(c.items())))

    if len(raters) >= 2:
        (na, ra), (nb, rb) = list(raters.items())[:2]
        both = [i for i in key if all(ra.get(i, {}).get(ax) and rb.get(i, {}).get(ax) for ax in AXES)]
        print(f"\n=== inter-rater ({na} vs {nb}), {len(both)} items scored by both ===")
        for ax in AXES:
            a = [ra[i][ax] for i in both]; b = [rb[i][ax] for i in both]
            agree = sum(x == y for x, y in zip(a, b)) / max(len(both), 1)
            print(f"  {ax:10s} agreement {agree*100:.0f}%   kappa {kappa(a, b)}")


if __name__ == "__main__":
    main()
