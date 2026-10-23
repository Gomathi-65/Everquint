"""Command line search + summary.   python main.py "oil prices opec" --n 5 --length short"""
import argparse

from search import HybridSearcher
from summarize import LENGTHS, summarize

ap = argparse.ArgumentParser()
ap.add_argument("query")
ap.add_argument("--n", type=int, default=5, help="number of documents to return")
ap.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
ap.add_argument("--length", default="medium", help="short | medium | long | number of words")
ap.add_argument("--provider", help="anthropic | openai (default: extractive, no key needed)")
a = ap.parse_args()

searcher = HybridSearcher.load_or_build()
results = searcher.search(a.query, a.n, a.mode)
for r in results:
    print(f"{r['rank']}. {r['title']}  (score {r['score']:.4f}, id {r['id']})\n   {r['excerpt'][:240]}\n")
if results:
    length = a.length if a.length in LENGTHS else int(a.length)
    s = summarize(results, a.query, length, a.provider)
    print(f"--- Summary ({s['method']}, {s['words']} words, {s['seconds']}s) ---\n{s['summary']}")
else:
    print("No results.")
