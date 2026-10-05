"""Does the k passed to search() change which neighbour comes back? TQ+ index, 3 seeds.
Usage: python3 k_dependence.py <slug> <bits>"""
import numpy as np, sys
from turbovec import TurboQuantIndex
slug, bw = sys.argv[1], int(sys.argv[2])
X=np.load(f"emb/{slug}.npy").astype(np.float32); X/=np.linalg.norm(X,axis=1,keepdims=True); d=X.shape[1]
for seed in (42,43,44):
    idx=np.random.RandomState(seed).permutation(len(X)); q,db=X[idx[:1000]],X[idx[1000:]]
    t=TurboQuantIndex(d,bit_width=bw); t.calibrate(db[np.random.RandomState(seed).choice(len(db),1024,replace=False)]); t.add(db)
    best=np.concatenate([(q[i:i+100]@db.T).max(1) for i in range(0,1000,100)])
    row=[]
    for k in (1,10,64):
        P=np.array(t.search(q,k=k)[1]); sc=np.einsum("qd,qkd->qk",q,db[P]); hit=sc>=best[:,None]-1e-6
        row.append(f"k={k}: R@1 {hit[:,0].mean():.3f} NN-in-top-{min(k,10)} {hit[:,:10].any(1).mean():.3f}")
    print(slug,bw,seed," | ".join(row), flush=True)
