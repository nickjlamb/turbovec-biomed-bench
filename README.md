# turbovec on biomedical embeddings

These are benchmark scripts and results for [turbovec](https://github.com/RyanCodrai/turbovec) 1.1.1 on PubMed embeddings (NCBI's MedCPT, d=768). The README's own OpenAI/DBpedia data (d=1536) serves as a reference.

The main question is the one turbovec's `docs/api.md` asks users to check: on data unlike the documented corpora, does the staged (4-bit) and two-stage (2-bit) search return the same ids as the whole-index scan?

## Results

### Agreement: staged vs whole-index scan

TQ+ index, 100K database, 3 seeds × 1,000 held-out queries:

| data | bits | identical id set, k=1 / 10 / 100 | recall10@10, staged / full |
|---|---|---|---|
| OpenAI d=1536 (DBpedia) | 4 | 100% / 100% / 100% | 0.969 / 0.969 |
| OpenAI d=1536 (DBpedia) | 2 | 100% / 100% / 100% | 0.898 / 0.898 |
| MedCPT d=768 (PubMed) | 4 | 99.2% / 82.6% / 57.2% | 0.931 / 0.944 |
| MedCPT d=768 (PubMed) | 2 | 99.6% / 83.1% / 79.1% | 0.775 / 0.779 |

### Recall@1

Measured at k=64, where the staged search is close to a full scan:

| data | bits | TQ | TQ+ | TQ+ on mean-centred docs | FAISS IndexPQ |
|---|---|---|---|---|---|
| OpenAI d=1536 | 4 | 0.973 | 0.970 | 0.973 | 0.964 |
| OpenAI d=1536 | 2 | 0.881 | 0.896 | 0.891 | 0.873 |
| MedCPT d=768 | 4 | 0.898 | 0.936 | 0.895 | 0.883 |
| MedCPT d=768 | 2 | 0.682 | 0.737 | 0.680 | 0.653 |

### Embedding geometry

| | mean pairwise cos | ‖mean of unit vectors‖ | median top-1 − top-2 gap |
|---|---|---|---|
| OpenAI d=1536 | 0.086 | 0.29 | 0.0266 |
| MedCPT d=768 | 0.647 | 0.80 | 0.0069 |

### Other findings

- **Centring helps agreement.** Mean-centring the documents leaves the exact ranking unchanged. It raises 4-bit k=10 agreement on MedCPT from 82.6% to 93.0%, but lowers recall@1 (table above).
- **The result depends on k.** For corrigendum PMID 36114062, the article it corrects (PMID 36115732) is the exact nearest neighbour. `search(k=64)` returns it at rank 1; `search(k=8)` and `search(k=10)` do not return it at all.
- **Duplicates.** 605 of the 101K MedCPT rows are exact duplicates. They are mostly errata and corrections notices, non-English records and identically titled letters. They give 1.0–1.4% of queries a tied top-1, so all hit tests here are tie-aware.
- **Small-model control.** On 11K PubMed abstracts, bge-small-en-v1.5 and its biomedical fine-tune MedEmbed-small behave almost identically. bge-small is also anisotropic on DBpedia text (mean pairwise cos 0.43).

### Follow-up: bge-small on the same articles

At the maintainer's request ([turbovec#562](https://github.com/RyanCodrai/turbovec/issues/562)), I embedded the same 101K PubMed articles with BAAI/bge-small-en-v1.5 (d=384). I used the title + abstract text NCBI fed MedCPT, truncated to 256 tokens. TQ+ index, staged vs whole-index scan:

| corpus | bits | identical ids, k=1 / 10 / 100 | recall10@10, staged / full |
|---|---|---|---|
| bge-small d=384 | 4 | 99.2% / 75.6% / 32.2% | 0.917 / 0.936 |
| bge-small d=384 | 2 | 99.2% / 69.0% / 47.5% | 0.767 / 0.776 |

Scripts: `embed_bge_medcpt_slice.py` (set `MEDCPT_DIR` to the folder holding NCBI's `pubmed_chunk_{30,36}.json`) and `sign_depth.py`, an approximation of the sign-bit depth statistic that uses a random rotation, so read its numbers as relative only. Results: `results/agree_bge-small-pubmed-101k_*bit.json` and `results/sign_depth_*.json`.

Raw numbers are in [`results/`](results/).

## Method

The harness follows turbovec's `benchmarks/suite/recall_*.py`:

- L2-normalise the vectors, then hold out 1,000 queries from the database, over 3 split seeds (42–44).
- TQ+ is `calibrate()` on 1,024 database rows.
- FAISS is `IndexPQ(nbits=8)` with m = d/4 at 2-bit and m = d/2 at 4-bit.

Two departures from the suite:

- **Tie-aware hits.** A returned id counts as the nearest neighbour if its exact score is within 1e-6 of the best. MedCPT contains exact duplicate vectors, so an argmax ground truth would score some correct answers as misses.
- **Agreement.** The staged and full-scan results come from separate processes. The full-scan process sets `TURBOVEC_2BIT_PLANES=0` / `TURBOVEC_4BIT_PLANES=0`, which are read once per process. turbovec's rotation is deterministic, and the OpenAI control returning 100% identical ids confirms the two processes build the same index.

The staged path only runs on indexes of at least 32,768 vectors. On x86 it also needs AVX-512 VBMI and VNNI. These runs used a 2-core x86 machine with both.

## Data

- **MedCPT.** [NCBI's precomputed PubMed embeddings](https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/). Two 50,500-row slices, from chunk 30 (PMIDs ~30.10M) and chunk 36 (PMIDs ~36.10M), fetched with HTTP range reads (~310 MB, not 3 GB per chunk).
- **OpenAI.** The first three parquet shards of [Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M), the dataset turbovec's README uses.
- **PubMed abstracts** for the small-model control: about 23K records with abstracts from 2015–2024, fetched via NCBI E-utilities and sampled down to 11K.

## Reproduce

Run the steps in order. Each step fits in about 7 GB of RAM.

```bash
pip install -r requirements.txt

# data
python3 fetch_medcpt.py                      # emb/medcpt-ncbi-100k.npy + PMIDs
python3 fetch_openai.py                      # emb/openai-1536-101k.npy

# recall: TQ / TQ+ / TQ+ centred / FAISS
python3 bench2.py medcpt-ncbi-100k openai-1536-101k

# staged vs whole-index agreement
for s in medcpt-ncbi-100k openai-1536-101k; do for b in 4 2; do
  python3 agreement.py $s $b staged
  TURBOVEC_2BIT_PLANES=0 TURBOVEC_4BIT_PLANES=0 python3 agreement.py $s $b full
  python3 agreement.py $s $b compare
done; done

# supporting checks
python3 corrigendum_case.py
python3 centring_agreement.py staged && TURBOVEC_4BIT_PLANES=0 python3 centring_agreement.py full && python3 centring_agreement.py compare
python3 duplicates.py
python3 k_dependence.py medcpt-ncbi-100k 4
python3 anisotropy_dbpedia.py

# small-model control (slow on CPU: ~15 min per model on 2 cores)
python3 fetch_pubmed.py 21000
N_DOCS=11000 python3 embed.py bge-small medembed-small
python3 bench2.py bge-small medembed-small
```

On a 2-core machine the full set takes a couple of hours, most of it embedding and FAISS training. The agreement check alone takes about 15 minutes.

| script | what it does |
|---|---|
| `bench2.py` | Recall@1@k and recall10@10 for TQ, TQ+, TQ+ on centred docs and FAISS IndexPQ, plus geometry stats |
| `agreement.py` | Staged vs whole-index id agreement and recall, in the api.md style |
| `centring_agreement.py` | Agreement with mean-centred documents |
| `corrigendum_case.py` | The k=8/10 vs k=64 example |
| `duplicates.py` | Duplicate rows and tied-top-1 queries |
| `k_dependence.py` | Whether the k passed to `search()` changes the top-1 |
| `anisotropy_dbpedia.py` | bge-small geometry on DBpedia text |
| `fetch_*.py`, `embed.py` | Data |

## Licence

MIT. MedCPT embeddings are from NCBI ([MedCPT](https://github.com/ncbi/MedCPT)). PubMed records are from the US National Library of Medicine.
