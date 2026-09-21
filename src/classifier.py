"""
Loads the trained classifier and predicts top-k probable conditions
from a structured symptom list.
"""


def load_model(path: str = "models/classifier.pkl"):
    """TODO: load trained model (joblib/pickle)."""
    raise NotImplementedError


def predict(symptoms: list[str], top_k: int = 3):
    """TODO: return top-k (condition, confidence) pairs."""
    raise NotImplementedError
