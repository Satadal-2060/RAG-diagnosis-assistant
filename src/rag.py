"""
RAG retrieval module: given a predicted condition, retrieves grounded
explanations from a trusted medical knowledge base using FAISS, then
synthesizes a natural-language explanation via the Groq API.
"""
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
embedder = SentenceTransformer('all-MiniLM-L6-v2')

NAME_FIXES = {
    "Hypertension ": "Hypertension",
    "Diabetes ": "Diabetes",
    "Dimorphic hemmorhoids(piles)": "Dimorphic hemorrhoids(piles)",
}


def classifier_to_corpus_name(classifier_name: str) -> str:
    return NAME_FIXES.get(classifier_name, classifier_name.strip())


def build_corpus(clf_classes, desc_df: pd.DataFrame, precaution_df: pd.DataFrame) -> dict:
    """Builds one text document per disease class from description + precaution data."""
    desc_df = desc_df.copy()
    precaution_df = precaution_df.copy()
    desc_df['Disease'] = desc_df['Disease'].str.strip()
    precaution_df['Disease'] = precaution_df['Disease'].str.strip()

    def build_doc(disease_name):
        corpus_name = classifier_to_corpus_name(disease_name)
        desc_row = desc_df[desc_df['Disease'] == corpus_name]
        prec_row = precaution_df[precaution_df['Disease'] == corpus_name]

        description = desc_row['Description'].values[0] if not desc_row.empty else "No description available."

        precautions = []
        if not prec_row.empty:
            for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
                val = prec_row[col].values[0]
                if pd.notna(val):
                    precautions.append(str(val))
        precaution_text = "; ".join(precautions) if precautions else "No specific precautions listed."

        return f"Disease: {corpus_name}\nDescription: {description}\nRecommended precautions: {precaution_text}"

    return {disease: build_doc(disease) for disease in clf_classes}


def build_index(corpus_documents: dict):
    """Builds a FAISS index over the corpus documents. Returns (index, disease_names, documents)."""
    disease_names = list(corpus_documents.keys())
    documents = list(corpus_documents.values())

    embeddings = embedder.encode(documents, show_progress_bar=False)
    embeddings = np.array(embeddings).astype('float32')

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, disease_names, documents


def retrieve_explanation_source(predicted_disease: str, index, documents: list, k: int = 1) -> str:
    """Retrieves the most relevant corpus document for a predicted disease."""
    query_embedding = embedder.encode([predicted_disease]).astype('float32')
    distances, indices = index.search(query_embedding, k)
    return documents[indices[0][0]]


def generate_explanation(predicted_disease: str, index, documents: list) -> str:
    """Retrieves grounded content and synthesizes a natural explanation via Groq."""
    retrieved_content = retrieve_explanation_source(predicted_disease, index, documents)

    system_prompt = """You are a medical information assistant. You will be given
retrieved, trusted source content about a specific medical condition. Write a clear,
natural-language explanation for a general audience using ONLY the information provided
in the retrieved content. Do not add any facts, statistics, or claims that are not
present in the retrieved content. Keep it concise (3-5 sentences). End with the
precautions mentioned, phrased naturally."""

    user_prompt = f"""Retrieved content:
{retrieved_content}

Write a natural explanation of this condition for someone who has been predicted to
possibly have it, based strictly on the retrieved content above."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()
