import html
import time
import pandas as pd
import streamlit as st
from src.extraction import extract_symptoms
from src.classifier import load_model, predict

st.set_page_config(page_title="SymptoSense", page_icon="🩺", layout="centered")

# ----------------------------------------------------------------------------
# PIPELINE CONNECTION & RESOURCE LOADING
# ----------------------------------------------------------------------------
CONF_THRESHOLD = 0.30  # fixed at 30%

# Load classifier and symptom vocabulary
clf = load_model("models/classifier.pkl")
train_df = pd.read_csv("data/Training.csv")
train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
symptom_vocab = [col for col in train_df.columns if col != 'prognosis']

# Wrapper around predict() so run_pipeline can call classify_symptoms
def classify_symptoms(symptoms):
    return predict(clf, symptoms, symptom_vocab)

def run_pipeline(text: str):
    symptoms = extract_symptoms(text,symptom_vocab)
    predictions = classify_symptoms(symptoms)

    if predictions and predictions[0]["confidence"] < CONF_THRESHOLD:
        explanation = (
            "The model has low confidence in its predictions. "
            "Please provide more detailed symptoms or consult a doctor."
        )
    else:
        explanation = (
            f"The top prediction is {predictions[0]['disease']} "
            f"with confidence {predictions[0]['confidence']*100:.1f}%. "
            "This condition is often associated with the listed symptoms."
        )

    return {
        "symptoms": symptoms,
        "predictions": predictions,
        "explanation": explanation,
    }

# ----------------------------------------------------------------------------
# Initialize session history
# ----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# ----------------------------------------------------------------------------
# Sidebar history view
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("🗂️ Chat History")
    if st.session_state["history"]:
        for item in reversed(st.session_state["history"]):
            st.markdown(f"**You:** {html.escape(item['query'])}")
            st.markdown(f"**Assistant:** {html.escape(item['result']['explanation'])}")
            st.markdown("---")
    else:
        st.caption("No history yet.")

# ----------------------------------------------------------------------------
# SEARCH BAR
# ----------------------------------------------------------------------------
with st.form("search"):
    query = st.text_area(
        "🔎 Describe your symptoms",
        placeholder="e.g. I have yellowish skin, dark urine and pain in my upper right abdomen...",
        height=110,
    )
    go = st.form_submit_button("Analyze Symptoms ✨")

if go:
    if not query.strip():
        st.warning("Please describe your symptoms first.")
    else:
        slot = st.empty()
        slot.markdown("⏳ Analyzing...", unsafe_allow_html=True)
        try:
            result = run_pipeline(query.strip())
        except Exception as e:
            slot.empty()
            st.error(f"Something went wrong in the pipeline: {e}")
            st.stop()
        slot.empty()

        # Save to session history
        st.session_state["history"].append({"query": query.strip(), "result": result})

        # Symptoms
        chips = "".join(
            f'<span class="chip">{html.escape(s.replace("_", " "))}</span>'
            for s in result["symptoms"]
        )
        st.markdown(f'<div class="card"><h3>🧾 Symptoms understood</h3>{chips}</div>', unsafe_allow_html=True)

        # Predictions
        preds = result["predictions"]
        if preds and preds[0]["confidence"] < CONF_THRESHOLD:
            st.markdown(
                '<div class="warn">⚠️ <b>Low confidence.</b> Add more specific symptoms or consult a doctor.</div>',
                unsafe_allow_html=True)
        bars = "".join(
            f'<div class="pred"><div class="n"><span>{"🥇" if i == 0 else "▫"} {html.escape(p["disease"])}</span>'
            f'<span>{p["confidence"]*100:.1f}%</span></div>'
            f'<div class="track"><div class="fill" style="--w:{p["confidence"]*100:.1f}%"></div></div></div>'
            for i, p in enumerate(preds)
        )
        st.markdown(f'<div class="card"><h3>📊 Possible conditions</h3>{bars}</div>', unsafe_allow_html=True)

        # Explanation
        st.markdown(
            f'<div class="card"><h3>📚 Grounded explanation</h3>{html.escape(result["explanation"])}</div>',
            unsafe_allow_html=True)

st.markdown(
    '<div class="disc">🛡️ SymptoSense is an educational triage assistant, not a medical diagnosis. '
    "Always consult a qualified doctor.</div>", unsafe_allow_html=True)

