"""Approximate Ryan's statistic: how deep in a pure sign-bit ranking do the exact top-10 sit?
Random orthogonal rotation (stand-in for turbovec's), asymmetric score = (Rq) . sign(Rx).
64 self-queries over the corpus; depth = worst rank, in the sign ranking, of the exact top-10 (self excluded)."""
import sys, json, numpy as np
slug = sys.argv[1]
X = np.load(f"emb/{slug}.npy").astype(np.float32)[:100_000]; X /= np.linalg.norm(X, axis=1, keepdims=True)
d = X.shape[1]; rng = np.random.RandomState(0)
R, _ = np.linalg.qr(rng.randn(d, d).astype(np.float32))
S = np.sign(X @ R).astype(np.float32)
qi = rng.choice(len(X), 64, replace=False); depths = []
for i in qi:
    exact = X @ X[i]; exact[i] = -9
    top10 = np.argpartition(-exact, 10)[:10]
    sb = S @ (X[i] @ R); sb[i] = -1e9
    order = np.argsort(-sb); rank = np.empty(len(X), int); rank[order] = np.arange(1, len(X) + 1)
    depths.append(int(rank[top10].max()))
d_ = np.array(depths); r = dict(corpus=slug, median=int(np.median(d_)), p90=int(np.percentile(d_, 90)), max=int(d_.max()))
print(json.dumps(r)); json.dump(r, open(f"results/sign_depth_{slug}.json", "w"))
