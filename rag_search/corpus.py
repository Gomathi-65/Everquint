"""Corpus loading and preprocessing (Reuters-21578 news, ~10.8k articles, via NLTK)."""
import hashlib
import html
import json
import re
from functools import lru_cache
from pathlib import Path

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

DATA_DIR = Path(__file__).parent / "data"
_stem = lru_cache(maxsize=None)(PorterStemmer().stem)
_WORD = re.compile(r"[a-z0-9]+")
MIN_WORDS = 30


def tokenize(text):
    """Lowercase, split on non-alphanumerics, drop stop words and 1-char tokens, Porter-stem."""
    return [_stem(w) for w in _WORD.findall(text.lower()) if len(w) > 1 and w not in ENGLISH_STOP_WORDS]


def clean_text(s):
    s = html.unescape(s).replace("\x03", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"\s*Reuter$", "", s).strip()


def load_corpus(rebuild=False):
    """Return a list of {id, title, text, categories}. Cached in data/corpus.jsonl."""
    cache = DATA_DIR / "corpus.jsonl"
    if cache.exists() and not rebuild:
        with cache.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    import nltk
    nltk.download("reuters", quiet=True)
    from nltk.corpus import reuters

    docs, seen, dropped = [], set(), {"short": 0, "duplicate": 0}
    for fid in reuters.fileids():
        head, _, body = reuters.raw(fid).strip().partition("\n")
        title, text = clean_text(head).title(), clean_text(body)
        if len(text.split()) < MIN_WORDS:
            dropped["short"] += 1
            continue
        digest = hashlib.md5(text.lower().encode()).hexdigest()
        if digest in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(digest)
        docs.append({"id": fid, "title": title, "text": text, "categories": reuters.categories(fid)})
    DATA_DIR.mkdir(exist_ok=True)
    with cache.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Corpus: kept {len(docs)} docs, dropped {dropped}")
    return docs
