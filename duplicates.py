"""Exact-duplicate vectors in the MedCPT slice, and how many held-out queries have a tied top-1."""
import json, collections, numpy as np
X = np.load("emb/medcpt-ncbi-100k.npy"); X /= np.linalg.norm(X, axis=1, keepdims=True)
dup_rows = int(len(X) - len(np.unique(np.round(X, 5), axis=0))); ties = {}
for seed in (42, 43, 44):
    idx = np.random.RandomState(seed).permutation(len(X)); q, db = X[idx[:1000]], X[idx[1000:]]; n = 0
    for i in range(0, 1000, 100):
        S = -np.partition(-(q[i:i+100] @ db.T), 2, 1)[:, :2]; S.sort(1); n += int(np.sum(S[:, 1] - S[:, 0] < 1e-6))
    ties[seed] = n / 1000
r = dict(duplicate_rows=dup_rows, queries_with_tied_top1=ties); json.dump(r, open("results/duplicates.json", "w"), indent=2); print(r)
