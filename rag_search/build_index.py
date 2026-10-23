"""Download/clean the corpus and build the search index (run once; takes a minute or two)."""
import time

from search import HybridSearcher

t = time.perf_counter()
s = HybridSearcher.load_or_build(rebuild=True)
print(f"Indexed {len(s.docs)} documents with BM25 + {s.dense_name} in {time.perf_counter() - t:.0f}s")
