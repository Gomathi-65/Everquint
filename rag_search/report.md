# Report: Document Search and Summarization

## 1. Data and pre-processing

**Corpus.** Reuters-21578 news articles (10,788 documents) loaded through NLTK. It is a well-known, freely available news corpus with enough documents to make search non-trivial and with headlines that serve as a human-written reference.

| Step | What | Why |
|---|---|---|
| Split headline and body | First line is the headline, rest is the body | Headline is indexed and used later as a reference summary |
| Clean text | Unescape HTML entities, remove control characters, collapse whitespace, strip the trailing "Reuter" signature | Noise hurts both term matching and embeddings |
| Drop short documents | Removed 2,140 articles under 30 words | Many are one-line notices that carry no content to search or summarize |
| De-duplicate | Removed 117 exact duplicates (MD5 of the normalised body) | Duplicates inflate results and make evaluation ambiguous |
| Tokenize for BM25 / TF-IDF | Lowercase, alphanumeric tokens, remove stop words and 1-character tokens, Porter stemming | Stemming lets "cuts" match "cut". Side effect: abbreviations such as "U.S." are lost |
| Headline weight | Headline text counted twice in the BM25 document | Headlines are dense summaries of the article |

Result: **8,531 documents**.

## 2. Methodology

**Search.** Two retrievers combined:
- **BM25** (`rank-bm25`) for exact and stemmed term matching.
- **Dense embeddings** for semantic matching. The code uses `sentence-transformers/all-MiniLM-L6-v2` when it is installed. The results below were produced in an environment without that package, so the dense side was **LSA** (TF-IDF + truncated SVD, 256 dimensions). LSA is a weaker semantic model than a neural sentence encoder.
- **Fusion:** reciprocal rank fusion, `score = sum(weight / (60 + rank))` over the top 100 of each ranking. BM25 has weight 1. The dense weight (0.1) was chosen on a separate dev sample (section 3.3). Each result includes the two sentences that best overlap with the query as an excerpt.

**Summarization.** For the top-K retrieved documents, with a user-chosen length (short ~50, medium ~120, long ~250 words, or a custom number):
- **LLM path** (Anthropic or OpenAI): the articles and query are sent with a prompt that asks for one coherent summary of about the target length, using only facts from the articles. If the call fails, it falls back to extractive.
- **Extractive path** (no key needed): sentences are scored by centrality (similarity to the TF-IDF centroid), query relevance and a small lead bonus; near-duplicate sentences are skipped; chosen sentences are emitted in document order until the word budget is reached.

**Efficiency and scalability.** A search takes about 25-35 ms on 8.5k documents. The index is cached on disk. For much larger corpora: replace `rank-bm25` (which scores every document) with an inverted-index engine (Elasticsearch/OpenSearch, Lucene/Pyserini), store embeddings in a vector index (FAISS or a vector database) with approximate nearest-neighbour search, encode documents in batches on a GPU, and summarize only the top K documents to bound LLM cost.

## 3. Evaluation

### 3.1 Retrieval

**Procedure.** A seeded random sample of 200 documents with at least 80 words. For each, a query that should return it, in two types:
- **keyword:** the 4 most distinctive body words (TF-IDF), excluding words from the headline.
- **headline:** the document's own headline.

A hit means the source document is in the top k. Corpus size 8,531, test size 200. Numbers are from a run on the author's Windows laptop (Python 3.13); an earlier run in another environment differed by at most 0.005 on keyword queries (one query in 200), probably due to library versions.

**Keyword queries**

| Mode | Hit@1 | Hit@5 | Hit@10 | MRR@10 | ms/query |
|---|---|---|---|---|---|
| BM25 | 0.905 | 0.995 | 0.995 | 0.946 | 30 |
| Dense (LSA) | 0.380 | 0.690 | 0.810 | 0.510 | 30 |
| Hybrid (dense weight 0.1) | 0.865 | 0.995 | 0.995 | 0.926 | 29 |

**Headline queries**

| Mode | Hit@1 | Hit@5 | Hit@10 | MRR@10 | ms/query |
|---|---|---|---|---|---|
| BM25 | 0.915 | 0.995 | 1.000 | 0.952 | 32 |
| Dense (LSA) | 0.510 | 0.730 | 0.815 | 0.615 | 32 |
| Hybrid (dense weight 0.1) | 0.845 | 0.995 | 1.000 | 0.907 | 32 |

### 3.2 Reading these results honestly

- **BM25 is the best single retriever on these queries.** The queries are built from words that occur in the target article, which is exactly what lexical search rewards. They contain no paraphrases.
- **The dense side (LSA) is much weaker here** and any weight on it lowered accuracy slightly. Hybrid is therefore a little below BM25 on these test queries, not above it.
- The benefit of dense retrieval is for queries that do not share words with the article ("central bank raises borrowing costs" for an article about an interest rate rise). This test set does not measure that. `python evaluate.py --llm-queries` generates natural-language queries with an LLM, which is the fairer test, and a neural encoder (`sentence-transformers`) should be much stronger than LSA. **Both were not run for this report.**
- Reuters contains near-duplicate stories (not removed beyond exact duplicates), so a "miss" can be a different but equally relevant article.

### 3.3 Choosing the fusion weight

Tuned on a dev sample (200 different documents, no overlap with the test set), keyword queries:

| Dense weight | Hit@1 | MRR@10 |
|---|---|---|
| 0.00 (BM25 only) | 0.915 | 0.948 |
| 0.10 | 0.885 | 0.932 |
| 0.30 | 0.815 | 0.888 |
| 0.50 | 0.765 | 0.854 |
| 1.00 (equal weight) | 0.705 | 0.800 |

I kept a small weight of 0.1 so that the dense ranking can still contribute semantic matches, accepting about 3 points of Hit@1 on lexical queries. With a neural encoder this should be re-tuned with `tune_weight.py`.

### 3.4 Summaries

No gold summaries exist for Reuters, so the **headline is the reference** (a human-written one-line summary), and ROUGE **recall** is reported for 50-word summaries of 60 documents, next to two baselines.

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg words |
|---|---|---|---|---|
| Extractive (ours) | 0.688 | 0.234 | 0.577 | 49.8 |
| Lead sentences | 0.666 | 0.232 | 0.568 | 64.8 |
| Random sentences | 0.536 | 0.155 | 0.463 | 66.4 |

Limits of this measure: recall against a very short reference favours longer summaries (the baselines are longer than ours), headline words usually appear in the first sentence (which favours lead-based methods), and ROUGE does not measure coherence or factual accuracy. The extractive summarizer is slightly better than the lead baseline with fewer words, which is a small but real signal.

**Human evaluation.** `results/human_eval.csv` holds 10 queries with their top results and a generated summary, plus empty columns for coherence, coverage and faithfulness (1-5). **This has not been filled in**: it needs a person to rate it. The same sheet should be generated with an LLM summarizer and compared with the extractive one.

## 4. Challenges and solutions

| Challenge | Solution |
|---|---|
| Hybrid scored below BM25 at first (equal weights, weak LSA side) | Added a fusion weight and tuned it on a separate dev set instead of the test set |
| No gold summaries for ROUGE | Used the headline as a reference and added lead and random baselines, and stated the limits |
| Generated test queries can be biased toward the retrieval method | Excluded headline words from keyword queries, added a second query type, and added an LLM-written query option |
| Slow start for the UI | Cache the whole index to disk; Streamlit caches it in memory |
| Redundant sentences in multi-document extractive summaries | Similarity filter between chosen sentences |
| Short or failing LLM calls | Automatic fallback to the extractive summarizer |

## 5. Not done / next steps

- Run with `sentence-transformers`, re-tune the weight, and evaluate with LLM-written queries.
- Fill in the human evaluation sheet and compare LLM and extractive summaries.
- Fine-tune a domain model (suggested in the task hints); not attempted.
- Hosted version of the UI: not deployed. It runs locally with `streamlit run app.py`.