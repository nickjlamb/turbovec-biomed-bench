"""Staged search vs whole-index scan: id-set agreement and recall, per docs/api.md's tables.

The api.md tables report "queries returning exactly the whole-index scan's ids" for OpenAI
and all-mpnet-base-v2, and ask users to "check agreement on your own data". This runs that
check. The planes switch is read once per process, so each mode runs in its own process:

    python3 agreement.py <slug> <bits> staged     # default path (index >= 32,768 vectors)
    TURBOVEC_2BIT_PLANES=0 TURBOVEC_4BIT_PLANES=0 python3 agreement.py <slug> <bits> full
    python3 agreement.py <slug> <bits> compare

The rotation is deterministic (api.md), so rebuilding the index in each process gives
the same codes; the OpenAI control returning 100% identical ids confirms this.
TQ+ (calibrated) index, 3 split seeds x 1,000 held-out queries, k = 1, 10, 100.
"""
import json, sys, numpy as np

slug, bw, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
SEEDS, NQ, CALIB, KS = [42, 43, 44], 1000, 1024, [1, 10, 100]
out = f"results/agree_{slug}_{bw}bit"

if mode in ("staged", "full"):
    from turbovec import TurboQuantIndex
    X = np.load(f"emb/{slug}.npy").astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True); d = X.shape[1]
    ids = {}
    for seed in SEEDS:
        idx = np.random.RandomState(seed).permutation(len(X)); q, db = X[idx[:NQ]], X[idx[NQ:]]
        t = TurboQuantIndex(d, bit_width=bw)
        t.calibrate(db[np.random.RandomState(seed).choice(len(db), CALIB, replace=False)]); t.add(db)
        for k in KS: ids[f"{seed}_{k}"] = np.array(t.search(q, k=k)[1])
        if mode == "staged":  # exact float truth, tie-aware: store best score + per-k thresholds
            best = []
            for i in range(0, NQ, 100):
                S = q[i:i + 100] @ db.T; top = -np.partition(-S, 100, 1)[:, :100]; top.sort(1); best.append(top[:, ::-1])
            best = np.concatenate(best)
            ids[f"{seed}_truth"] = best  # descending exact top-100 scores
        for k in KS: ids[f"{seed}_{k}_scores"] = np.einsum("qd,qkd->qk", q, db[ids[f"{seed}_{k}"]])
    np.savez(f"{out}_{mode}.npz", **ids); print("saved", mode)
else:
    S, F = np.load(f"{out}_staged.npz"), np.load(f"{out}_full.npz")
    res = {}
    for k in KS:
        same = np.concatenate([[set(a) == set(b) for a, b in zip(S[f"{s}_{k}"], F[f"{s}_{k}"])] for s in SEEDS])
        shared = np.concatenate([[len(set(a) & set(b)) / k for a, b in zip(S[f"{s}_{k}"], F[f"{s}_{k}"])] for s in SEEDS])
        nn_staged = np.concatenate([(S[f"{s}_{k}_scores"] >= S[f"{s}_truth"][:, :1] - 1e-6).any(1) for s in SEEDS])
        nn_full = np.concatenate([(F[f"{s}_{k}_scores"] >= S[f"{s}_truth"][:, :1] - 1e-6).any(1) for s in SEEDS])
        same_scores = np.concatenate([np.all(np.abs(np.sort(S[f"{s}_{k}_scores"], 1) - np.sort(F[f"{s}_{k}_scores"], 1)) < 1e-6, 1) for s in SEEDS])
        rk_s = np.concatenate([(S[f"{s}_{k}_scores"] >= S[f"{s}_truth"][:, k-1:k] - 1e-6).sum(1) / k for s in SEEDS])
        rk_f = np.concatenate([(F[f"{s}_{k}_scores"] >= S[f"{s}_truth"][:, k-1:k] - 1e-6).sum(1) / k for s in SEEDS])
        per_seed = [round(float(np.mean([set(a) == set(b) for a, b in zip(S[f"{s}_{k}"], F[f"{s}_{k}"])])), 4) for s in SEEDS]
        res[f"k={k}"] = dict(identical_id_set=round(float(same.mean()), 4), identical_id_set_per_seed=per_seed,
                             identical_score_set_tol_1e6=round(float(same_scores.mean()), 4),
                             recall_k_at_k_staged=round(float(rk_s.mean()), 4), recall_k_at_k_full_scan=round(float(rk_f.mean()), 4),
                             ids_shared=round(float(shared.mean()), 4),
                             true_nn_in_topk_staged=round(float(nn_staged.mean()), 4),
                             true_nn_in_topk_full_scan=round(float(nn_full.mean()), 4))
    json.dump(res, open(f"{out}.json", "w"), indent=2); print(slug, bw, json.dumps(res))
