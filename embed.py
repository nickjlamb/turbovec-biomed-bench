"""Embed pubmed.jsonl with each model; writes emb/<slug>.npy (float32, raw, not normalised).

All models see the same text (title + abstract) truncated to 256 tokens, on CPU.
"""
import json, os, sys, time, numpy as np, torch

MODELS = {
    # slug: (hf id, kind, note)
    "bge-small":     ("BAAI/bge-small-en-v1.5",                 "st",     "general, 384-d"),
    "medembed-small":("abhinand/MedEmbed-small-v0.1",           "st",     "biomedical fine-tune of bge-small, 384-d"),
    "mpnet-base":    ("sentence-transformers/all-mpnet-base-v2", "st",     "general, 768-d"),
    "pubmedbert":    ("NeuML/pubmedbert-base-embeddings",       "st",     "biomedical, 768-d"),
    "medcpt":        ("ncbi/MedCPT-Article-Encoder",             "medcpt", "biomedical retriever, 768-d, CLS"),
}
MAX_LEN = 256
torch.set_num_threads(os.cpu_count())

recs = [json.loads(l) for l in open("pubmed.jsonl")]
N = int(os.environ.get("N_DOCS", "11000"))
recs = [recs[i] for i in np.random.RandomState(0).permutation(len(recs))[:N]]  # sample across all years
os.makedirs("emb", exist_ok=True)
json.dump([r["pmid"] for r in recs], open("emb/pmids.json", "w"))

for slug in (sys.argv[1:] or MODELS):
    hf, kind, _ = MODELS[slug]
    out = f"emb/{slug}.npy"
    if os.path.exists(out): print("skip", slug); continue
    t0 = time.time()
    if kind == "st":
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(hf, device="cpu"); m.max_seq_length = MAX_LEN
        texts = [f"{r['title']} {r['abstract']}" for r in recs]
        E = m.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
    else:  # MedCPT article encoder: [title, abstract] pairs, CLS token
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(hf); mdl = AutoModel.from_pretrained(hf).eval()
        chunks = []
        with torch.no_grad():
            for i in range(0, len(recs), 32):
                b = recs[i:i+32]
                enc = tok([[r["title"], r["abstract"]] for r in b], truncation=True, padding=True,
                          return_tensors="pt", max_length=MAX_LEN)
                chunks.append(mdl(**enc).last_hidden_state[:, 0, :].numpy())
        E = np.concatenate(chunks)
    np.save(out, E.astype(np.float32))
    print(f"{slug}: {E.shape} in {time.time()-t0:.0f}s", flush=True)
