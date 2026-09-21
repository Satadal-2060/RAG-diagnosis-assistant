"""
Symptom extraction module: parses free-text user input into a structured
symptom list using the Groq API, matching the classifier's expected vocabulary.
"""
import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def extract_symptoms(text: str) -> list[str]:
    """
    Takes free-text symptom description, returns a structured list of
    symptoms matching the classifier's known symptom vocabulary.
    TODO: implement prompt + parsing logic.
    """
    raise NotImplementedError
