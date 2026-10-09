
import html
import time
import streamlit as st

st.set_page_config(page_title="SymptoSense", page_icon="🩺", layout="centered")

# ----------------------------------------------------------------------------
# BACKEND CONNECTION
# ----------------------------------------------------------------------------
CONF_THRESHOLD = 0.30  # still fixed at 30%

def run_pipeline(text: str):
    try:
        from backend import diagnose
        return diagnose(text), False
    except ImportError:
        time.sleep(2)
        return {
            "symptoms": ["yellowish_skin", "dark_urine", "abdominal_pain"],
            "predictions": [
                {"disease": "Jaundice", "confidence": 0.345},
                {"disease": "Hepatitis B", "confidence": 0.21},
                {"disease": "Chronic cholestasis", "confidence": 0.15},
            ],
            "explanation": "Demo explanation: Jaundice is caused by high bilirubin levels...",
        }, True

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
            result, demo = run_pipeline(query.strip())
        except Exception as e:
            slot.empty()
            st.error(f"Something went wrong in the backend: {e}")
            st.stop()
        slot.empty()

        # Save to session history
        st.session_state["history"].append({"query": query.strip(), "result": result})

        if demo:
            st.markdown('<div class="demo">⚠️ Demo mode: backend.py not found, showing sample output</div>',
                        unsafe_allow_html=True)

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
            f'<div class="pred"><div class="n"><span>{"🥇" if i == 0 else "▫️"} {html.escape(p["disease"])}</span>'
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
