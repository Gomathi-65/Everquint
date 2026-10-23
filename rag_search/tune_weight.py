"""Pick the dense weight for hybrid fusion on a DEV sample (different seed, no overlap with the test set)."""
import random

import numpy as np

from evaluate import keyword_queries
from search import HybridSearcher

s = HybridSearcher.load_or_build()
docs = s.docs
eligible = [i for i, d in enumerate(docs) if len(d["text"].split()) >= 80]
test = set(random.Random(42).sample(eligible, 200))
dev = random.Random(7).sample([i for i in eligible if i not in test], 200)
queries = keyword_queries(docs, dev)
print("dense_weight  hit@1  mrr@10   (dev set, keyword queries)")
for w in [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
    ranks = []
    for i, q in queries.items():
        res = s.search(q, 10, "hybrid", dense_weight=w)
        ranks.append(next((r["rank"] for r in res if r["index"] == i), None))
    print(f"{w:12.2f} {np.mean([r == 1 for r in ranks]):7.3f} {np.mean([1 / r if r else 0 for r in ranks]):7.3f}")
