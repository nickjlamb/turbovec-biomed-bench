"""Recall of turbovec (TQ, TQ+, TQ+ on mean-centred docs) vs FAISS IndexPQ on biomedical embeddings.

Methodology follows turbovec/benchmarks/suite/recall_*.py:
  - L2-normalise; random split into database + 1,000 held-out queries (3 split seeds)
  - recall@1@k: is the true nearest neighbour inside the returned top-k?
  - TQ+ = calibrate() on a 1,024-row database sample
  - FAISS IndexPQ(nbits=8), m = d/4 at 2-bit, m = d/2 at 4-bit (bit-rate matched)
One change: the hit test is TIE-AWARE. Real corpora contain exact duplicate vectors
(PubMed: ~0.6% of MedCPT rows), so argmax picks one of several equally-correct
answers arbitrarily. A returned id counts as the nearest neighbour if its exact
score equals the best exact score (within 1e-6). recall10@10 likewise counts returned
ids whose exact score reaches the true 10th-best score.

Centring: for unit-norm docs x and query q, q.(x - mu) = q.x - q.mu, and q.mu is a
per-query constant, so exact ranking is unchanged. Docs are centred (not re-normalised),
queries untouched, mu estimated from the calibration sample.
"""
import json, sys, time, numpy as np, faiss
from turbovec import TurboQuantIndex

K = 64; K_VALUES = [1, 2, 4, 8, 16, 32, 64]; NQ = 1000; CALIB = 1024; SEEDS = [42, 43, 44]; EPS = 1e-6


def geometry(X, rng):
    s = X[rng.choice(len(X), 2000, replace=False)]
    C = s @ s.T; iu = np.triu_indices(len(s), 1)
    ev = np.linalg.eigvalsh(np.cov(X[rng.choice(len(X), min(len(X), 20000), replace=False)].T))[::-1]
    return dict(mean_pairwise_cos=round(float(C[iu].mean()), 4),
                participation_ratio=round(float(ev.sum() ** 2 / (ev ** 2).sum()), 1),
                norm_of_mean_vector=round(float(np.linalg.norm(X.mean(0))), 4),
                exact_duplicate_rows=int(len(X) - len(np.unique(np.round(X, 5), axis=0))))


def metrics(q, db, best, tenth, P):
    sc = np.einsum("qd,qkd->qk", q, db[P])  # exact scores of the returned ids
    hit = sc >= best[:, None] - EPS
    out = {f"r1@{k}": float(hit[:, :k].any(1).mean()) for k in K_VALUES}
    out["r10@10"] = float((sc[:, :10] >= tenth[:, None] - EPS).sum(1).mean() / 10)
    return out


def run(slug):
    X = np.load(f"emb/{slug}.npy").astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True)
    d = X.shape[1]
    res = {"model": slug, "dim": d, "database_size": len(X) - NQ, "geometry": geometry(X, np.random.RandomState(0))}
    gaps = []
    for bw in (2, 4):
        m = d // 4 if bw == 2 else d // 2
        rows = {}
        for seed in SEEDS:
            idx = np.random.RandomState(seed).permutation(len(X)); q, db = X[idx[:NQ]], X[idx[NQ:]]
            best, tenth = [], []
            for i in range(0, NQ, 100):
                S = q[i:i + 100] @ db.T
                top = -np.partition(-S, 10, 1)[:, :10]; top.sort(1)  # ascending: top[:, -1] best, top[:, 0] 10th
                best.append(top[:, -1]); tenth.append(top[:, 0])
                if bw == 2: gaps.append(top[:, -1] - top[:, -2])
            best, tenth = np.concatenate(best), np.concatenate(tenth)
            samp = db[np.random.RandomState(seed).choice(len(db), CALIB, replace=False)]
            mu = samp.mean(0).astype(np.float32)
            P = {}
            t = TurboQuantIndex(d, bit_width=bw); t.add(db); P["tq"] = np.array(t.search(q, k=K)[1]); del t
            t = TurboQuantIndex(d, bit_width=bw); t.calibrate(samp); t.add(db); P["tq+"] = np.array(t.search(q, k=K)[1]); del t
            t = TurboQuantIndex(d, bit_width=bw); t.calibrate(np.ascontiguousarray(samp - mu))
            t.add(np.ascontiguousarray(db - mu)); P["tq+ centred"] = np.array(t.search(q, k=K)[1]); del t
            f = faiss.IndexPQ(d, m, 8, faiss.METRIC_INNER_PRODUCT); f.train(db); f.add(db)
            P["faiss"] = f.search(q, K)[1]; del f
            for name, p in P.items():
                rows.setdefault(name, []).append(metrics(q, db, best, tenth, p))
        res[f"{bw}bit"] = {n: {k: [round(float(np.mean([r[k] for r in v])), 4), round(float(np.std([r[k] for r in v])), 4)]
                               for k in v[0]} for n, v in rows.items()}
        s = res[f"{bw}bit"]
        print(f"{slug} d={d} {bw}-bit R@1 " + "  ".join(f"{n} {s[n]['r1@1'][0]:.3f}" for n in s)
              + f" | R1@8 tq+ {s['tq+']['r1@8'][0]:.4f} faiss {s['faiss']['r1@8'][0]:.4f}", flush=True)
    res["median_top1_top2_gap"] = round(float(np.median(np.concatenate(gaps))), 5)
    return res


if __name__ == "__main__":
    for slug in sys.argv[1:]:
        t0 = time.time(); r = run(slug)
        print(f"  geometry {r['geometry']} gap {r['median_top1_top2_gap']} ({time.time() - t0:.0f}s)", flush=True)
        json.dump(r, open(f"results/v2_{slug}.json", "w"), indent=2)
