"""Pull 3 of the 26 parquet shards of the README's own dataset (Qdrant DBpedia, OpenAI
text-embedding-3-large d=1536) -> 101K rows, to check this harness reproduces README recall."""
import os, urllib.request, pyarrow.parquet as pq, numpy as np, io
B = "https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M/resolve/main/data/"
os.makedirs("emb", exist_ok=True)
parts = []
for i in range(3):
    raw = urllib.request.urlopen(f"{B}train-{i:05d}-of-00026.parquet", timeout=900).read()
    t = pq.read_table(io.BytesIO(raw), columns=["text-embedding-3-large-1536-embedding"])
    parts.append(np.stack(t.column(0).to_numpy(zero_copy_only=False)).astype(np.float32)); del raw, t; print(i, parts[-1].shape, flush=True)
X = np.concatenate(parts)[:101_000]; np.save("emb/openai-1536-101k.npy", X); print("saved", X.shape)
