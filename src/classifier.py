"""
Loads the trained classifier and predicts top-k probable conditions
from a structured symptom list, with confidence-aware messaging.
"""
import joblib
import pandas as pd


def load_model(path: str = "models/classifier.pkl"):
    return joblib.load(path)


def symptoms_to_vector(extracted_symptoms: list[str], symptom_vocab: list[str]) -> pd.DataFrame:
    vector = {col: (1 if col in extracted_symptoms else 0) for col in symptom_vocab}
    return pd.DataFrame([vector], columns=symptom_vocab)


def predict(clf, extracted_symptoms: list[str], symptom_vocab: list[str], top_k: int = 3, confidence_threshold: float = 0.30):
    if not extracted_symptoms:
        return {
            "predictions": [],
            "message": "No recognizable symptoms found in the input. Please describe your symptoms more specifically."
        }

    X = symptoms_to_vector(extracted_symptoms, symptom_vocab)
    probs = clf.predict_proba(X)[0]
    top_indices = probs.argsort()[::-1][:top_k]

    predictions = [
        {"condition": clf.classes_[i], "confidence": round(float(probs[i]), 4)}
        for i in top_indices
    ]

    message = None
    if predictions[0]["confidence"] < confidence_threshold:
        message = (
            "The described symptoms are common to several conditions, so this prediction "
            "has lower confidence. Consider providing more specific or additional symptoms "
            "for a more reliable result."
        )

    return {"predictions": predictions, "message": message}
