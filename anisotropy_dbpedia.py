# Is the anisotropy from the model or from PubMed text? Embed 1,500 DBpedia texts (the README's corpus) with bge-small.
import urllib.request, io, numpy as np, pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
raw = urllib.request.urlopen("https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M/resolve/main/data/train-00000-of-00026.parquet", timeout=900).read()
t = pq.read_table(io.BytesIO(raw), columns=["title", "text"]); del raw
texts = [f"{a} {b}" for a, b in zip(t.column(0).to_pylist()[:1500], t.column(1).to_pylist()[:1500])]
m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu"); m.max_seq_length = 256
E = m.encode(texts, batch_size=32, normalize_embeddings=True)
C = E @ E.T; iu = np.triu_indices(len(E), 1)
r = dict(model="BAAI/bge-small-en-v1.5", corpus="DBpedia (README dataset), first 1,500 texts", mean_pairwise_cos=round(float(C[iu].mean()), 4), norm_of_mean=round(float(np.linalg.norm(E.mean(0))), 4)); import json; json.dump(r, open("results/anisotropy_bge_dbpedia.json", "w"), indent=2); print(r)
