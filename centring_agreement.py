"""Does subtracting the corpus mean (docs only; ranking-preserving) change staged/full agreement?
MedCPT, 4-bit TQ+, k=10, 3 seeds. Run 'staged', then 'full' with TURBOVEC_4BIT_PLANES=0, then 'compare'."""
import sys, json, numpy as np
mode = sys.argv[1]; SEEDS = [42, 43, 44]
if mode != "compare":
    from turbovec import TurboQuantIndex
    X = np.load("emb/medcpt-ncbi-100k.npy"); X /= np.linalg.norm(X, axis=1, keepdims=True); out = {}
    for seed in SEEDS:
        idx = np.random.RandomState(seed).permutation(len(X)); q, db = X[idx[:1000]], X[idx[1000:]]
        s = db[np.random.RandomState(seed).choice(len(db), 1024, replace=False)]; mu = s.mean(0).astype(np.float32)
        t = TurboQuantIndex(768, bit_width=4); t.calibrate(np.ascontiguousarray(s - mu)); t.add(np.ascontiguousarray(db - mu))
        out[str(seed)] = np.array(t.search(q, k=10)[1])
    np.savez(f"results/centring_agree_{mode}.npz", **out); print("saved", mode)
else:
    a, b = np.load("results/centring_agree_staged.npz"), np.load("results/centring_agree_full.npz")
    per = {s: round(float(np.mean([set(x) == set(y) for x, y in zip(a[s], b[s])])), 4) for s in a.files}
    r = {"centred_identical_id_set_k10_per_seed": per, "mean": round(float(np.mean(list(per.values()))), 4),
         "uncentred_reference": "results/agree_medcpt-ncbi-100k_4bit.json k=10"}
    json.dump(r, open("results/centring_agreement.json", "w"), indent=2); print(r)
