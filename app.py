import html
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="SymptoSense", page_icon="60A", layout="centered")

# ----------------------------------------------------------------------------
# API KEY (must be set BEFORE importing src.extraction / src.rag, because both
# create the Groq client at import time)
# ----------------------------------------------------------------------------
load_dotenv()  # local runs: reads .env
try:
    if "GROQ_API_KEY" in st.secrets:  # Streamlit Cloud: app Settings -> Secrets
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass  # no secrets file locally

if not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Add it to Streamlit Secrets (cloud) or .env (local).")
    st.stop()

from src.extraction import extract_symptoms
from src.classifier import load_model, predict
from src.rag import (
    build_corpus,
    build_index,
    generate_explanation,
    retrieve_explanation_source,
)

# ----------------------------------------------------------------------------
# STYLES (your HTML uses these classes, so they must be defined somewhere)
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
.card{border:1px solid rgba(128,128,128,.25);border-radius:14px;padding:1rem 1.2rem;
      margin:.8rem 0;background:rgba(128,128,128,.06)}
.card h3{margin:0 0 .6rem 0;font-size:1.05rem}
.chip{display:inline-block;padding:.2rem .65rem;margin:.15rem .25rem .15rem 0;border-radius:999px;
      background:rgba(46,134,222,.15);border:1px solid rgba(46,134,222,.4);font-size:.85rem}
.warn{border-left:4px solid #f5a623;background:rgba(245,166,35,.12);padding:.7rem 1rem;
      border-radius:8px;margin:.8rem 0}
.pred{margin:.6rem 0}
.pred .n{display:flex;justify-content:space-between;font-size:.95rem;margin-bottom:.25rem}
.track{height:8px;border-radius:999px;background:rgba(128,128,128,.25);overflow:hidden}
.fill{height:100%;width:var(--w);border-radius:999px;
      background:linear-gradient(90deg,#2e86de,#1abc9c);animation:grow .8s ease-out}
@keyframes grow{from{width:0}to{width:var(--w)}}
.disc{margin-top:1.5rem;font-size:.8rem;opacity:.7;text-align:center}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# PIPELINE CONNECTION & RESOURCE LOADING
# ----------------------------------------------------------------------------
CONF_THRESHOLD = 0.30  # fixed at 30%
# ASSUMPTION: knowledge-base files live in data/rag_corpus/ under these names.
DESC_PATH = "data/rag_corpus/symptom_Description.csv"
PREC_PATH = "data/rag_corpus/symptom_precaution.csv"


@st.cache_resource(show_spinner="Loading model and knowledge base...")
def load_resources():
    clf = load_model("models/classifier.pkl")
    header = pd.read_csv("data/Training.csv", nrows=0)
    vocab = [c for c in header.columns if c != "prognosis" and not c.startswith("Unnamed")]
    corpus = build_corpus(clf.classes_, pd.read_csv(DESC_PATH), pd.read_csv(PREC_PATH))
    index, _names, documents = build_index(corpus)
    return clf, vocab, index, documents


clf, symptom_vocab, index, documents = load_resources()


@st.cache_data(show_spinner=False)
def cached_explanation(disease: str, _index, _documents) -> str:
    # leading underscores tell Streamlit not to hash the index/documents
    return generate_explanation(disease, _index, _documents)


def run_pipeline(text: str):
    symptoms = extract_symptoms(text, symptom_vocab)
    out = predict(
        clf, symptoms, symptom_vocab, top_k=3, confidence_threshold=CONF_THRESHOLD
    )  # -> {"predictions": [{"condition", "confidence"}], "message": str | None}
    preds = out["predictions"]

    explanation, source = None, None
    if preds and not out["message"]:  # only explain confident predictions
        top = str(preds[0]["condition"])
        explanation = cached_explanation(top, index, documents)
        source = retrieve_explanation_source(top, index, documents)

    return {
        "symptoms": symptoms,
        "predictions": preds,
        "message": out["message"],
        "explanation": explanation,
        "source": source,
    }


def render_result(result):
    # Symptoms
    chips = "".join(
        f'<span class="chip">{html.escape(s.replace("_", " "))}</span>'
        for s in result["symptoms"]
    ) or "<i>No known symptoms recognized.</i>"
    st.markdown(f'<div class="card"><h3>ǴCD Symptoms understood</h3>{chips}</div>',
                unsafe_allow_html=True)

    # Low-confidence / no-symptom message from predict()
    if result["message"]:
        st.markdown(f'<div class="warn">⚠️ {html.escape(result["message"])}</div>',
                    unsafe_allow_html=True)

    # Predictions
    preds = result["predictions"]
    if preds:
        bars = "".join(
            f'<div class="pred"><div class="n">'
            f'<span>{"🥇" if i == 0 else "▫"} {html.escape(str(p["condition"]))}</span>'
            f'<span>{p["confidence"] * 100:.1f}%</span></div>'
            f'<div class="track"><div class="fill" style="--w:{p["confidence"] * 100:.1f}%"></div></div></div>'
            for i, p in enumerate(preds)
        )
        st.markdown(f'<div class="card"><h3>ǴCA Possible conditions</h3>{bars}</div>',
                    unsafe_allow_html=True)

    # Grounded explanation (RAG)
    if result["explanation"]:
        body = html.escape(result["explanation"]).replace("\n", "<br>")
        st.markdown(f'<div class="card"><h3>ǴD6 Grounded explanation</h3>{body}</div>',
                    unsafe_allow_html=True)
        with st.expander("Source used for this explanation"):
            st.text(result["source"])


# ----------------------------------------------------------------------------
# Session history
# ----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# ----------------------------------------------------------------------------
# MAIN UI
# ----------------------------------------------------------------------------
st.title("Ǵ8C SymptoSense")

with st.form("search"):
    query = st.text_area(
        "ǵ0E Describe your symptoms",
        placeholder="e.g. I have yellowish skin, dark urine and pain in my upper right abdomen...",
        height=110,
    )
    go = st.form_submit_button("Analyze Symptoms ✨")

if go:
    if not query.strip():
        st.warning("Please describe your symptoms first.")
    else:
        try:
            with st.spinner("Analyzing..."):
                result = run_pipeline(query.strip())
        except Exception as e:
            st.error("Something went wrong in the pipeline.")
            st.exception(e)  # remove once debugging is done
        else:
            st.session_state["history"].append({
                "query": query.strip(),
                "summary": result["explanation"] or result["message"] or "No result.",
            })
            render_result(result)

st.markdown(
    '<div class="disc">ǶE1️ SymptoSense is an educational triage assistant, not a medical diagnosis. '
    "Always consult a qualified doctor.</div>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar history. Rendered LAST so the entry added above shows up immediately
# (in your version the sidebar was drawn before the new item existed, so it
# always lagged one run behind).
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("ǴC2 Chat History")
    if st.session_state["history"]:
        for item in reversed(st.session_state["history"]):
            st.markdown(f"**You:** {html.escape(item['query'])}")
            st.markdown(f"**Assistant:** {html.escape(item['summary'])}")
            st.markdown("---")
    else:
        st.caption("No history yet.")
