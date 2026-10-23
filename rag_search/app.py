"""Streamlit UI:  streamlit run app.py"""
import streamlit as st

from search import HybridSearcher
from summarize import LENGTHS, summarize

PAGE_SIZE = 5
st.set_page_config(page_title="Document Search & Summary", layout="wide")


@st.cache_resource(show_spinner="Loading index (first run builds it, about a minute)...")
def load():
    return HybridSearcher.load_or_build()


searcher = load()
st.title("Document Search and Summarization")
st.caption(f"{len(searcher.docs):,} Reuters news articles | BM25 + {searcher.dense_name}")

with st.sidebar:
    mode = st.radio("Search mode", ["hybrid", "bm25", "dense"], help="hybrid = BM25 + embeddings fused with reciprocal rank fusion")
    top_n = st.slider("Documents to retrieve (top N)", 5, 50, 15)
    choice = st.radio("Summary length", ["short", "medium", "long", "custom"], index=1, horizontal=True)
    words = st.slider("Words", 20, 400, 100) if choice == "custom" else LENGTHS[choice]
    summ_k = st.slider("Summarize the top K documents", 1, 10, 5)
    provider = st.selectbox("Summarizer", ["extractive (no key)", "anthropic", "openai"])

if "q" not in st.session_state:
    st.session_state.q = ""
st.text_input("Search", key="q", placeholder="e.g. opec oil production cut")
for s in searcher.suggest(st.session_state.q):
    st.button(s, key="sug_" + s, on_click=lambda s=s: st.session_state.update(q=s))

query = st.session_state.q.strip()
if query:
    results = searcher.search(query, top_n, mode)
    if not results:
        st.warning("No results. Try other words.")
        st.stop()
    prov = None if provider.startswith("extractive") else provider
    s = summarize(results[:summ_k], query, words, prov)
    st.subheader("Summary")
    st.write(s["summary"])
    st.caption(f"{s['method']} | {s['words']} words (target {s['target_words']}) | {s['seconds']}s | from top {min(summ_k, len(results))} documents")

    st.subheader(f"Results ({len(results)})")
    pages = max(1, -(-len(results) // PAGE_SIZE))
    page = st.session_state.get("page", 1)
    page = min(page, pages)
    for r in results[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]:
        with st.container(border=True):
            st.markdown(f"**{r['rank']}. {r['title']}**  \n`{r['id']}` | score {r['score']:.4f} | {', '.join(r['categories'])}")
            st.write(r["excerpt"])
            with st.expander("Full text"):
                st.write(r["text"])
    c1, c2, c3 = st.columns([1, 1, 6])
    if c1.button("Previous", disabled=page <= 1):
        st.session_state.page = page - 1
        st.rerun()
    if c2.button("Next", disabled=page >= pages):
        st.session_state.page = page + 1
        st.rerun()
    c3.write(f"Page {page} of {pages}")
