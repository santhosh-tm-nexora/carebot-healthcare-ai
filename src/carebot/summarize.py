"""Summary stage. Uses an LLM when ANTHROPIC_API_KEY is set, else a template fallback.
The LLM is constrained to the extracted entities to limit hallucination."""
import os

PROMPT = """You summarize clinical notes for patient records.
Write ONE short paragraph using ONLY facts in the transcript and entities below.
Do not add diagnoses, drugs, doses or dates that are not present.

Transcript:
{text}

Extracted entities:
{entities}
"""


def template_summary(e: dict) -> str:
    parts = []
    if e["diagnosis"]:
        parts.append("Diagnosis: " + ", ".join(e["diagnosis"]) + ".")
    if e["medications"]:
        parts.append("Medications: " + ", ".join(
            f'{m["name"]} {m["dosage"]}'.strip() if m["dosage"] else m["name"] for m in e["medications"]) + ".")
    if e["follow_up"]:
        parts.append("Follow-up: " + " ".join(e["follow_up"]))
    return " ".join(parts) or "No structured findings extracted."


def summarize(text: str, entities: dict, model: str = "claude-sonnet-4-5") -> str:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return template_summary(entities)
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model, max_tokens=300,
        messages=[{"role": "user", "content": PROMPT.format(text=text, entities=entities)}],
    )
    return msg.content[0].text.strip()
