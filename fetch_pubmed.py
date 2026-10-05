"""Fetch ~21K PubMed abstracts (title + abstract) via NCBI E-utilities.

Sampling: PubMed records with abstracts, English, journal articles, taken from
Entrez-date windows spread across 2015-2024 so the corpus spans specialties
and years rather than one topic. Output: pubmed.jsonl  {pmid, title, abstract}
"""
import json, time, sys, xml.etree.ElementTree as ET, urllib.parse, urllib.request

EU = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 21_000
YEARS = range(2015, 2025)
PER_YEAR = TARGET // len(YEARS) + 200  # headroom for records that fail parsing

def get(url, params):
    for attempt in range(5):
        try:
            q = urllib.parse.urlencode(params)
            with urllib.request.urlopen(f"{EU}{url}?{q}", timeout=60) as r:
                return r.read()
        except Exception as e:
            time.sleep(2 * (attempt + 1)); err = e
    raise err

def pmids_for_year(y, n):
    term = (f'hasabstract[text] AND english[lang] AND journal article[pt] '
            f'AND {y}/03/01:{y}/03/31[edat]')
    x = ET.fromstring(get("esearch.fcgi", {"db": "pubmed", "term": term, "retmax": n, "sort": "pub_date"}))
    return [e.text for e in x.findall(".//Id")]

def fetch(pmids):
    x = ET.fromstring(get("efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}))
    out = []
    for art in x.findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID")
        title = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
        abst = " ".join("".join(a.itertext()) for a in art.findall(".//Abstract/AbstractText"))
        if len(abst) > 200:
            out.append({"pmid": pmid, "title": title.strip(), "abstract": abst.strip()})
    return out

seen, n = set(), 0
with open("pubmed.jsonl", "w") as f:
    for y in YEARS:
        ids = pmids_for_year(y, PER_YEAR); time.sleep(0.4)
        for i in range(0, len(ids), 200):
            for rec in fetch(ids[i:i + 200]):
                if rec["pmid"] not in seen:
                    seen.add(rec["pmid"]); f.write(json.dumps(rec) + "\n"); n += 1
            time.sleep(0.4)
        print(y, len(ids), "total", n, flush=True)
print("done", n)
