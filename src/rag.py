"""
RAG retrieval module: given a predicted condition, retrieves grounded
explanations from a trusted medical knowledge base using FAISS.
"""


def build_index(corpus_path: str):
    """TODO: build/load the FAISS index from the medical corpus."""
    raise NotImplementedError


def retrieve_explanation(condition: str, k: int = 3):
    """TODO: retrieve top-k relevant passages and synthesize an explanation."""
    raise NotImplementedError
