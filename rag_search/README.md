# Document Search and Summarization

Hybrid document search (BM25 + embeddings) over ~8.5k Reuters news articles, with length-controlled summaries, an evaluation harness and a Streamlit UI. The full write-up is in [`report.md`](report.md).

## Setup and run

Python 3.9+.

```
pip install -r requirements.txt
python build_index.py                      # downloads the corpus via NLTK, cleans it, builds the index (~1 min)

python main.py "oil prices opec production cut" --n 5 --length short     # command line
streamlit run app.py                       # web UI
python evaluate.py                         # retrieval + summary evaluation -> results/
python tune_weight.py                      # choose the hybrid fusion weight on a dev set
```

No API key is needed: without one, summaries are extractive. To use an LLM for summaries (and optionally for writing the evaluation queries):

```
pip install anthropic                      # or: pip install openai
export ANTHROPIC_API_KEY=...               # PowerShell: $env:ANTHROPIC_API_KEY="..."
export LLM_PROVIDER=anthropic              # or openai
python main.py "opec output" --length long
python evaluate.py --llm-queries
```

For better semantic search install `sentence-transformers` and rebuild with `python build_index.py`. Without it, the dense side falls back to LSA (TF-IDF + SVD).

## Files

| File | Purpose |
|---|---|
| `corpus.py` | Load, clean, de-duplicate, tokenize (stop words + Porter stemming) |
| `search.py` | BM25 + dense embeddings, reciprocal-rank fusion, excerpts, query suggestions, index caching |
| `summarize.py` | LLM summaries (Anthropic/OpenAI) or extractive fallback; short / medium / long / custom words |
| `llm.py` | Provider clients |
| `evaluate.py` | Hit@k, MRR, latency, ROUGE, human evaluation sheet |
| `tune_weight.py` | Chooses the fusion weight on a dev sample |
| `app.py` | Streamlit UI: search modes, pagination, suggestions, adjustable summary length |
| `main.py`, `build_index.py` | CLI and index build |
| `results/` | `eval_results.json`, `human_eval.csv` |

## Features

- Top-N retrieval with excerpts, three modes (`bm25`, `dense`, `hybrid`).
- Summary length: short (~50 words), medium (~120), long (~250) or any number of words.
- UI: query auto-suggestion, pagination, adjustable summary length and top-K to summarize.
- The index is cached in `data/index.pkl`, so the UI starts fast after the first build.
