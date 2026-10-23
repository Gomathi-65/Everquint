"""Evaluation: retrieval accuracy (Hit@k, MRR) and summary quality (ROUGE + human sheet).

    python evaluate.py                 # 200 test documents, keyword + headline queries
    python evaluate.py --llm-queries   # let an LLM write the queries (needs a key, see llm.py)

Test set: a seeded random sample of documents (>= 80 words). For each one a query is generated that
should return it. Query types:
  keyword  - the 4 most distinctive body words (TF-IDF), headline words excluded
  headline - the document's headline used as the query
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer

from llm import get_llm
from search import HybridSearcher
from summarize import _sentences, extractive_summary

OUT = Path(__file__).parent / "results"
MODES = ["bm25", "dense", "hybrid"]


def keyword_queries(docs, test_ids, k=4):
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2, max_df=0.3,
                          token_pattern=r"(?u)\b[a-zA-Z]{4,}\b")
    X = vec.fit_transform([d["text"] for d in docs])
    terms = np.array(vec.get_feature_names_out())
    out = {}
    for i in test_ids:
        title = set(docs[i]["title"].lower().split())
        row = X[i].toarray().ravel()
        top = [t for t in terms[np.argsort(-row)[:30]] if t not in title][:k]
        out[i] = " ".join(top)
    return out


def llm_queries(docs, test_ids, llm):
    out = {}
    for i in test_ids:
        prompt = ("Write a short search query (3-8 words) a user might type to find this article. "
                  "Do not copy the headline. Reply with the query only.\n\n" + docs[i]["text"][:1200])
        out[i] = llm.complete("You write realistic search queries.", prompt).strip().strip('"')
    return out


def retrieval_eval(searcher, queries, label):
    rows = {}
    for mode in MODES:
        ranks, lat = [], []
        for i, q in queries.items():
            t = time.perf_counter()
            res = searcher.search(q, 10, mode)
            lat.append(time.perf_counter() - t)
            pos = next((r["rank"] for r in res if r["index"] == i), None)
            ranks.append(pos)
        hit = lambda k: float(np.mean([p is not None and p <= k for p in ranks]))
        rows[mode] = {"hit@1": hit(1), "hit@5": hit(5), "hit@10": hit(10),
                      "mrr@10": float(np.mean([1 / p if p else 0 for p in ranks])),
                      "avg_ms": 1000 * float(np.mean(lat))}
    print(f"\n{label} queries")
    print(f"{'mode':8} {'hit@1':>6} {'hit@5':>6} {'hit@10':>7} {'mrr@10':>7} {'ms/query':>9}")
    for m, r in rows.items():
        print(f"{m:8} {r['hit@1']:6.3f} {r['hit@5']:6.3f} {r['hit@10']:7.3f} {r['mrr@10']:7.3f} {r['avg_ms']:9.1f}")
    return rows


def summary_eval(docs, test_ids, n=60, seed=42):
    """No gold summaries exist for Reuters, so the headline is the reference (a human-written
    one-line summary) and we report ROUGE *recall* against it, next to two baselines."""
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rng = random.Random(seed)
    ids = test_ids[:n]
    res = {name: {"rouge1": [], "rouge2": [], "rougeL": [], "words": []} for name in ["extractive", "lead", "random"]}
    for i in ids:
        d = docs[i]
        budget = 50
        sents = _sentences(d["text"])
        outputs = {"extractive": extractive_summary([d], d["title"], budget)}
        lead, used = [], 0
        for s in sents:
            if used >= budget:
                break
            lead.append(s)
            used += len(s.split())
        outputs["lead"] = " ".join(lead)
        pick, used = [], 0
        for s in rng.sample(sents, len(sents)):
            if used >= budget:
                break
            pick.append(s)
            used += len(s.split())
        outputs["random"] = " ".join(pick)
        for name, text in outputs.items():
            s = sc.score(d["title"], text)
            for k in ("rouge1", "rouge2", "rougeL"):
                res[name][k].append(s[k].recall)
            res[name]["words"].append(len(text.split()))
    final = {name: {k: float(np.mean(v)) for k, v in m.items()} for name, m in res.items()}
    print("\nSummary quality (ROUGE recall vs headline, 50-word summaries, %d docs)" % len(ids))
    print(f"{'method':11} {'R-1':>6} {'R-2':>6} {'R-L':>6} {'words':>6}")
    for name, m in final.items():
        print(f"{name:11} {m['rouge1']:6.3f} {m['rouge2']:6.3f} {m['rougeL']:6.3f} {m['words']:6.1f}")
    return final


def human_eval_sheet(searcher, queries, path, n=10):
    from summarize import summarize
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query", "top_doc_titles", "summary", "coherence_1to5", "coverage_1to5", "faithfulness_1to5", "notes"])
        for i, q in list(queries.items())[:n]:
            res = searcher.search(q, 3, "hybrid")
            s = summarize(res, q, "short")
            w.writerow([q, " | ".join(r["title"] for r in res), s["summary"], "", "", "", ""])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--llm-queries", action="store_true")
    ap.add_argument("--provider")
    a = ap.parse_args()
    searcher = HybridSearcher.load_or_build()
    docs = searcher.docs
    rng = random.Random(42)
    eligible = [i for i, d in enumerate(docs) if len(d["text"].split()) >= 80]
    test_ids = rng.sample(eligible, a.n)
    results = {"n_docs": len(docs), "n_test": len(test_ids), "dense": searcher.dense_name}
    kw = keyword_queries(docs, test_ids)
    results["keyword"] = retrieval_eval(searcher, kw, "Keyword")
    results["headline"] = retrieval_eval(searcher, {i: docs[i]["title"] for i in test_ids}, "Headline")
    if a.llm_queries:
        llm = get_llm(a.provider)
        if llm is None:
            raise SystemExit("--llm-queries needs --provider anthropic|openai (or LLM_PROVIDER)")
        results["llm"] = retrieval_eval(searcher, llm_queries(docs, test_ids, llm), "LLM-written")
    results["summary"] = summary_eval(docs, test_ids)
    OUT.mkdir(exist_ok=True)
    (OUT / "eval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    human_eval_sheet(searcher, kw, OUT / "human_eval.csv")
    print(f"\nSaved results/eval_results.json and results/human_eval.csv ({searcher.dense_name})")


if __name__ == "__main__":
    main()
