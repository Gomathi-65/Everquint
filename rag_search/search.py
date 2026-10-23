"""Hybrid search: BM25 (lexical) + dense embeddings (semantic), fused with reciprocal rank fusion."""
import pickle
import re
from collections import Counter

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from corpus import DATA_DIR, load_corpus, tokenize

RRF_K = 60
DENSE_WEIGHT = 0.1  # weight of the dense ranking in the fusion (BM25 weight is 1); chosen with tune_weight.py
ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _doc_text(d):
    return d["title"] + ". " + d["text"]


def _normalize(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(n, 1e-12)


def _top(scores, k):
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def best_excerpt(text, query, n=2):
    """The n sentences sharing the most (stemmed) terms with the query, in original order."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    q = set(tokenize(query))
    scored = [(len(q & set(tokenize(s))), -i, s) for i, s in enumerate(sents)]
    chosen = sorted(sorted(scored, reverse=True)[:n], key=lambda t: -t[1])
    return " ".join(s for _, _, s in chosen)


class HybridSearcher:
    def __init__(self, docs, embedder="auto"):
        self.docs = docs
        self.embedder = embedder
        self.tokens = [tokenize(d["title"] + " " + d["title"] + " " + d["text"]) for d in docs]  # title counted twice
        self.bm25 = BM25Okapi(self.tokens)
        self._build_dense()
        words = Counter(w for d in docs for w in re.findall(r"[a-z]{4,}", (d["title"] + " " + d["text"]).lower()))
        self.vocab = [w for w, c in words.most_common(6000) if c >= 15]
        self.titles = [d["title"] for d in docs]

    # ---------- dense side ----------
    def _build_dense(self):
        texts = [_doc_text(d) for d in self.docs]
        if self.embedder in ("auto", "st"):
            try:
                from sentence_transformers import SentenceTransformer
                self._st = SentenceTransformer(ST_MODEL)
                self.X = self._st.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
                self.dense_name = ST_MODEL
                return
            except Exception as e:
                if self.embedder == "st":
                    raise
                print(f"sentence-transformers unavailable ({type(e).__name__}); using LSA fallback")
        self.tfidf = TfidfVectorizer(analyzer=tokenize, sublinear_tf=True, min_df=2, max_df=0.5)
        svd = TruncatedSVD(n_components=256, random_state=42)
        self.X = _normalize(svd.fit_transform(self.tfidf.fit_transform(texts)))
        self.svd, self.dense_name = svd, "LSA (TF-IDF + SVD, 256 dims)"

    def _encode(self, query):
        if self.dense_name == ST_MODEL:
            if not hasattr(self, "_st"):
                from sentence_transformers import SentenceTransformer
                self._st = SentenceTransformer(ST_MODEL)
            return self._st.encode([query], normalize_embeddings=True)[0]
        return _normalize(self.svd.transform(self.tfidf.transform([_doc_text({"title": "", "text": query})])))[0]

    # ---------- search ----------
    def search(self, query, top_n=10, mode="hybrid", pool=100, dense_weight=None):
        """mode: 'bm25' | 'dense' | 'hybrid'. Returns ranked results with excerpts."""
        qt = tokenize(query)
        bm = self.bm25.get_scores(qt) if qt else np.zeros(len(self.docs))
        dn = self.X @ self._encode(query)
        if mode == "bm25":
            ids, scores = _top(bm, top_n), bm
        elif mode == "dense":
            ids, scores = _top(dn, top_n), dn
        else:
            fused = Counter()
            w_dn = DENSE_WEIGHT if dense_weight is None else dense_weight
            for s, w in ((bm, 1.0), (dn, w_dn)):
                for rank, i in enumerate(_top(s, pool), 1):
                    if s[i] > 0:
                        fused[i] += w / (RRF_K + rank)
            ids = [i for i, _ in fused.most_common(top_n)]
            scores = np.zeros(len(self.docs))
            for i, v in fused.items():
                scores[i] = v
        out = []
        for rank, i in enumerate(ids, 1):
            if scores[i] <= 0:
                break
            d = self.docs[int(i)]
            out.append({"rank": rank, "index": int(i), "id": d["id"], "title": d["title"], "score": float(scores[i]),
                        "excerpt": best_excerpt(d["text"], query), "text": d["text"], "categories": d["categories"]})
        return out

    def suggest(self, text, n=6):
        """Query auto-suggestion: complete the last word from the vocabulary, plus matching headlines."""
        text = text.lower().strip()
        if len(text) < 2:
            return []
        *head, last = text.split(" ")
        out = [" ".join(head + [w]) for w in self.vocab if w.startswith(last) and w != last][:n - 2]
        out += [t for t in self.titles if text in t.lower()][:2]
        return out[:n]

    # ---------- persistence ----------
    def save(self, path=None):
        path = path or DATA_DIR / "index.pkl"
        state = {k: v for k, v in self.__dict__.items() if k != "_st"}
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_or_build(cls, embedder="auto", rebuild=False):
        path = DATA_DIR / "index.pkl"
        if path.exists() and not rebuild:
            with open(path, "rb") as f:
                obj = cls.__new__(cls)
                obj.__dict__.update(pickle.load(f))
                return obj
        DATA_DIR.mkdir(exist_ok=True)
        obj = cls(load_corpus(rebuild), embedder)
        obj.save(path)
        return obj
