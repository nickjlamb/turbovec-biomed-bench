"""Range-read slices of NCBI's precomputed MedCPT PubMed embeddings (no full 3 GB download).

Two 50,500-row slices from different chunks (PMIDs ~30M and ~36M, i.e. different years)
-> 101,000 x 768 float32, matching the README's 100K database + 1K queries.
"""
import json, os, urllib.request, numpy as np, ast
os.makedirs("emb", exist_ok=True)
BASE = "https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/"
ROWS, D = 50_500, 768
parts = []
for chunk in (30, 36):
    url = f"{BASE}embeds_chunk_{chunk}.npy"
    head = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-511"})).read()
    hlen = int.from_bytes(head[8:10], "little"); off = 10 + hlen
    meta = ast.literal_eval(head[10:off].decode("latin1"))
    assert meta["descr"] == "<f4" and meta["shape"][1] == D, meta
    start = off + 100_000 * D * 4  # skip first 100K rows (oldest PMIDs in chunk)
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + ROWS*D*4 - 1}"})
    buf = urllib.request.urlopen(req, timeout=600).read()
    parts.append(np.frombuffer(buf, dtype="<f4").reshape(ROWS, D)); print(chunk, meta["shape"], flush=True)
np.save("emb/medcpt-ncbi-100k.npy", np.concatenate(parts))
pmids = []
for chunk in (30, 36):  # PMID list for the same rows
    p = json.load(urllib.request.urlopen(f"{BASE}pmids_chunk_{chunk}.json", timeout=600))
    pmids += [str(x) for x in p[100_000:100_000 + ROWS]]
json.dump(pmids, open("emb/medcpt-ncbi-100k.pmids.json", "w"))
print("saved")
