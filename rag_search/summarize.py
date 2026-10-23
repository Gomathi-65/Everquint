"""Summarize retrieved documents. LLM (Anthropic/OpenAI) when configured, otherwise extractive."""
import re
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from corpus import tokenize
from llm import get_llm

LENGTHS = {"short": 50, "medium": 120, "long": 250}  # target words

SUMMARY_SYSTEM = """You summarize news articles retrieved for a search query.
Write ONE coherent summary of about {words} words that captures the main points of the articles that are relevant to the query.
Use only facts stated in the articles. Do not invent numbers, names or dates. If articles disagree, say so briefly.
Plain prose, no bullet points, no preamble. Never go over {words} words by more than 10%."""


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 5]


def extractive_summary(docs, query, words):
    """Pick informative, non-redundant sentences across docs until the word budget is reached."""
    sents = [(di, si, s) for di, d in enumerate(docs) for si, s in enumerate(_sentences(d["text"]))]
    if not sents:
        return ""
    vec = TfidfVectorizer(analyzer=tokenize, sublinear_tf=True)
    X = vec.fit_transform([s for _, _, s in sents])
    centroid = np.asarray(X.mean(axis=0))
    centrality = np.asarray(X @ centroid.T).ravel()
    q = vec.transform([query])
    relevance = np.asarray((X @ q.T).todense()).ravel() if q.nnz else np.zeros(len(sents))
    lead = np.array([1.0 if si == 0 else 0.0 for _, si, _ in sents])
    score = centrality / (centrality.max() or 1) + 1.5 * relevance + 0.3 * lead
    chosen, used = [], 0
    for i in np.argsort(-score):
        n = len(sents[i][2].split())
        if used + n > words * 1.1 and chosen:
            continue
        if any((X[i] @ X[j].T).toarray()[0, 0] > 0.6 for j in chosen):  # redundancy filter
            continue
        chosen.append(int(i))
        used += n
        if used >= words:
            break
    return " ".join(sents[i][2] for i in sorted(chosen, key=lambda i: (sents[i][0], sents[i][1])))


def summarize(results, query, length="medium", provider=None, llm=None):
    """length: 'short' | 'medium' | 'long' or an int number of words."""
    words = length if isinstance(length, int) else LENGTHS[length]
    llm = llm or get_llm(provider)
    t0 = time.perf_counter()
    method = "extractive"
    if llm is not None:
        body = "\n\n".join(f"[{i}] {r['title']}\n{r['text'][:1500]}" for i, r in enumerate(results, 1))
        try:
            text = llm.complete(SUMMARY_SYSTEM.format(words=words), f"Query: {query}\n\nArticles:\n{body}").strip()
            method = type(llm).__name__
        except Exception as e:  # network/key problem: fall back instead of failing the search
            text = extractive_summary(results, query, words)
            method = f"extractive (LLM failed: {type(e).__name__})"
    else:
        text = extractive_summary(results, query, words)
    return {"summary": text, "method": method, "words": len(text.split()), "target_words": words,
            "seconds": round(time.perf_counter() - t0, 3)}
