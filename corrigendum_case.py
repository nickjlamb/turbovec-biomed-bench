"""The corrigendum example: same TQ+ 4-bit index (seed 42 split), searched at k=8 vs k=64."""
import json, numpy as np
from turbovec import TurboQuantIndex
X = np.load("emb/medcpt-ncbi-100k.npy"); X /= np.linalg.norm(X, axis=1, keepdims=True); pm = json.load(open("emb/medcpt-ncbi-100k.pmids.json"))
idx = np.random.RandomState(42).permutation(len(X)); qi, di = idx[:1000], idx[1000:]; q, db = X[qi], X[di]
t = TurboQuantIndex(768, bit_width=4); t.calibrate(db[np.random.RandomState(42).choice(len(db), 1024, replace=False)]); t.add(db)
i = [j for j in range(1000) if pm[qi[j]] == "36114062"][0]
exact = db @ q[i]; order = np.argsort(-exact)[:3]
r = {"query": "36114062", "exact_top3": [[pm[di[j]], round(float(exact[j]), 4)] for j in order]}
for k in (8, 10, 64):
    ids = [pm[di[j]] for j in np.array(t.search(q[i:i+1], k=k)[1])[0]]
    r[f"search_k{k}_top3"] = ids[:3]; r[f"search_k{k}_rank_of_36115732"] = ids.index("36115732") + 1 if "36115732" in ids else None
json.dump(r, open("results/corrigendum_case.json", "w"), indent=2); print(r)
