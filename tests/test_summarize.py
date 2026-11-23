from types import SimpleNamespace

from carebot.summarize import summarize, template_summary


ENTITIES = {
    "diagnosis": ["migraine"],
    "medications": [],
    "follow_up": [],
}


def test_missing_groq_configuration_uses_template(monkeypatch, capsys):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    assert summarize("clinical note", ENTITIES) == template_summary(ENTITIES)
    assert "GROQ_API_KEY" in capsys.readouterr().err


def test_groq_summary_uses_configured_model_and_base_url(monkeypatch):
    calls = {}

    class FakeCompletions:
        def create(self, **kwargs):
            calls["request"] = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="LLM summary"))]
            )

    class FakeClient:
        def __init__(self, **kwargs):
            calls["client"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    monkeypatch.setattr("openai.OpenAI", FakeClient)

    assert summarize("clinical note", ENTITIES) == "LLM summary"
    assert calls["client"] == {
        "api_key": "test-key",
        "base_url": "https://api.groq.com/openai/v1",
    }
    assert calls["request"]["model"] == "test-model"


def test_gpt_oss_uses_completion_budget_and_low_reasoning(monkeypatch):
    calls = {}

    class FakeCompletions:
        def create(self, **kwargs):
            calls["request"] = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="Entity-constrained summary"),
                    )
                ]
            )

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setattr("openai.OpenAI", FakeClient)

    assert summarize("clinical note", ENTITIES) == "Entity-constrained summary"
    assert calls["request"]["max_completion_tokens"] == 800
    assert calls["request"]["reasoning_effort"] == "low"


def test_groq_failure_uses_template_without_printing_key(monkeypatch, capsys):
    key = "test-secret-key"

    class FakeCompletions:
        def create(self, **kwargs):
            raise RuntimeError(f"request failed for {key}")

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("GROQ_API_KEY", key)
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    monkeypatch.setattr("openai.OpenAI", FakeClient)

    assert summarize("clinical note", ENTITIES) == template_summary(ENTITIES)
    error = capsys.readouterr().err
    assert "request failed" in error
    assert key not in error
    assert "[REDACTED]" in error
