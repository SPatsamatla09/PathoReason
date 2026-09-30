"""Operating characteristics of the pre-registered classification at K reps x 100 tiles.
True effect is exact: p'(t) = p(t) + D * (1 - p(t)) / mean(1 - p), so mean_t p' - p = D
(larger shifts on HP-leaning tiles, as an explain-first push toward SSA would be).
Tile base rates: 30% near-certain HP (0.03), 30% near-certain SSA (0.97), 40% U(0.15, 0.85).
The other three Holm family members are independent null p-values."""
import sys, random
sys.path.insert(0, '/Users/adityak/Documents/mhist/grid_experiment')
import analyze_ordering_controls as A
from collections import Counter
A.N_BOOT, A.N_PERM = 2000, 2000
K, R = int(sys.argv[1]), int(sys.argv[2])
tiles = [f't{i}' for i in range(100)]
def sim(D, seed):
    rng = random.Random(seed)
    base = {}
    for t in tiles:
        u = rng.random()
        base[t] = 0.03 if u < 0.3 else 0.97 if u < 0.6 else rng.uniform(0.15, 0.85)
    m = sum(1 - p for p in base.values()) / len(tiles)
    def draw(D):
        recs = {}
        for t in tiles:
            p = base[t] + D * (1 - base[t]) / m
            for r in range(1, K + 1):
                recs[(t, r)] = {'parsed': {'label': 'SSA' if rng.random() < p else 'HP'}}
        return {'recs': recs}
    co, c = draw(0), draw(D)
    reps = list(range(1, K + 1))
    dl = A.delta(co, c, tiles, reps, reps, seed)
    pv = {'x': dl['signflip_p'], **{f'n{j}': rng.random() for j in range(3)}}
    return A.classify(dl, A.holm_fixed(pv, 4)['x']).split(' (')[0]
for D in (0.0, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15):
    c = Counter(sim(D, 7919 * i + int(D * 1e4)) for i in range(R))
    print(f"K={K} trueD={D:.3f} R={R} " + " ".join(f"{k}={v/R:.3f}" for k, v in sorted(c.items())), flush=True)
