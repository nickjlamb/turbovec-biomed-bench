"""Embed the same 101K PubMed articles as the MedCPT slice with BAAI/bge-small-en-v1.5.

Text = the title + abstract NCBI fed MedCPT (pubmed_chunk_{30,36}.json, fields t and a), so the
MedCPT and bge corpora differ only in the model. 256 tokens, CPU, checkpointed every 1,000 rows.
Output: emb/bge-small-pubmed-101k.npy (raw, not normalised), rows aligned with medcpt-ncbi-100k.pmids.json.
"""
import json, os, sys, numpy as np, ijson
from sentence_transformers import SentenceTransformer

SRC = os.environ.get("MEDCPT_DIR", ".")  # folder holding NCBI pubmed_chunk_{30,36}.json
pmids = json.load(open("emb/medcpt-ncbi-100k.pmids.json")); want = set(pmids)
text_path = "emb/medcpt-slice-texts.json"
if not os.path.exists(text_path):
    text = {}
    for c in (30, 36):
        with open(f"{SRC}/pubmed_chunk_{c}.json", "rb") as f:
            for k, v in ijson.kvitems(f, ""):
                if k in want: text[k] = ((v.get("t") or "") + " " + (v.get("a") or "")).strip()
        print("chunk", c, "texts so far", len(text), flush=True)
    json.dump([text.get(p, "") for p in pmids], open(text_path, "w"))
texts = json.load(open(text_path))
print("missing/empty texts:", sum(1 for t in texts if not t), flush=True)

m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu"); m.max_seq_length = 256
os.makedirs("emb/bge_parts", exist_ok=True); STEP = 1000
for s in range(0, len(texts), STEP):
    part = f"emb/bge_parts/{s:06d}.npy"
    if os.path.exists(part): continue
    np.save(part, m.encode(texts[s:s+STEP], batch_size=32, convert_to_numpy=True, normalize_embeddings=False).astype(np.float32))
    print("done", s + STEP, flush=True)
E = np.concatenate([np.load(f"emb/bge_parts/{s:06d}.npy") for s in range(0, len(texts), STEP)])
assert len(E) == len(pmids); np.save("emb/bge-small-pubmed-101k.npy", E); print("saved", E.shape)
