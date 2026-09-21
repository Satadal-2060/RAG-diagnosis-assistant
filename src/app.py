"""
Streamlit entry point for the RAG-augmented symptom-to-diagnosis assistant.
Currently a placeholder - wire in extraction.py, classifier.py, and rag.py here.
"""
import streamlit as st

st.set_page_config(page_title="Symptom Assistant", page_icon="🩺")
st.title("Symptom-to-Diagnosis Assistant")
st.caption("Educational triage support - not a diagnostic authority. Consult a doctor.")

symptom_text = st.text_area("Describe your symptoms:")

if st.button("Analyze") and symptom_text.strip():
    st.info("Pipeline not yet wired up. Next: connect extraction -> classifier -> RAG here.")
