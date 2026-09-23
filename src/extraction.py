"""
Symptom extraction module: parses free-text user input into a structured
symptom list using the Groq API, matching the classifier's expected vocabulary.
"""
import os
import json
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def extract_symptoms(user_text: str, symptom_vocab: list[str]) -> list[str]:
    """
    Takes free-text symptom description, returns a structured list of
    symptoms matching the classifier's known symptom vocabulary.
    """
    system_prompt = f"""You are a medical symptom extractor. Given a user's free-text description of how they feel, identify which symptoms from the following fixed list are present.

Symptom list: {", ".join(symptom_vocab)}

Rules:
- Only return symptoms from this exact list, using their exact spelling.
- Only include symptoms clearly indicated by the text.
- Return ONLY a JSON array of matching symptom strings, nothing else. No explanation, no markdown, no preamble.
- If no symptoms match, return an empty array [].
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        symptoms = json.loads(raw)
        symptoms = [s for s in symptoms if s in symptom_vocab]
        return symptoms
    except json.JSONDecodeError:
        print("Failed to parse LLM output:", raw)
        return []
