"""Summary stage using Groq when configured, with a template fallback."""
import os
import sys

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


def _fallback(entities: dict, error: str) -> str:
    print(f"Groq summarization unavailable: {error}", file=sys.stderr)
    return template_summary(entities)


def _safe_error(error: Exception, api_key: str) -> str:
    return str(error).replace(api_key, "[REDACTED]")


def _extract_summary_content(response) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            else:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    text_parts.append(text)
        return "\n".join(text_parts).strip()

    refusal = getattr(message, "refusal", None)
    if isinstance(refusal, str) and refusal.strip():
        raise ValueError("Groq refused the summary request")
    return ""


def summarize(text: str, entities: dict) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL")
    missing = [
        name for name, value in (("GROQ_API_KEY", api_key), ("GROQ_MODEL", model))
        if not value
    ]
    if missing:
        return _fallback(
            entities,
            "missing environment variable(s): " + ", ".join(missing),
        )

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        request = {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT.format(text=text, entities=entities)}],
        }
        if model.startswith("openai/gpt-oss"):
            request["max_completion_tokens"] = 800
            request["reasoning_effort"] = "low"
        else:
            request["max_tokens"] = 300
        response = client.chat.completions.create(**request)
        content = _extract_summary_content(response)
        if not content:
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            raise ValueError(
                "Groq returned an empty summary"
                + (f" (finish reason: {finish_reason})" if finish_reason else "")
            )
        return content
    except Exception as error:
        return _fallback(entities, _safe_error(error, api_key))
