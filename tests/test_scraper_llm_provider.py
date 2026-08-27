"""Mocked tests for the commentary LLM provider boundary."""

from io import BytesIO
import json
import sys
from types import ModuleType, SimpleNamespace
from urllib.error import HTTPError

import pytest

from src.scraper import _llm_provider
from src.scraper._llm_errors import ProviderRequestError, provider_error_is_retryable
from src.scraper._llm_provider import LLMProvider, LLMSettings

TEST_SCHEMA: dict[str, object] = {
    "type": "array",
    "items": {"type": "string"},
}


class FakeResponse:
    """Minimal urlopen response for a non-streaming JSON envelope."""

    def __init__(self, payload: object) -> None:
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def _settings(provider: LLMProvider, key_environment: str) -> LLMSettings:
    return LLMSettings(
        provider=provider,
        model="provider/test-model",
        api_key_environment=key_environment,
    )


@pytest.mark.parametrize(
    "key_environment",
    ["sk-or-v1-DUMMY", "123_PROVIDER_KEY", "PROVIDER-KEY"],
)
def test_settings_reject_invalid_key_environment_names(key_environment: str) -> None:
    with pytest.raises(ValueError, match="portable environment variable name") as caught:
        _settings(LLMProvider.OPENROUTER, key_environment)

    assert key_environment not in str(caught.value)


def test_gemini_uses_native_sdk_with_json_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    google = ModuleType("google")
    genai = ModuleType("google.genai")

    def client(**kwargs: object) -> SimpleNamespace:
        captured["client"] = kwargs

        def generate_content(**request: object) -> SimpleNamespace:
            captured["request"] = request
            return SimpleNamespace(text='["ok"]')

        return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

    genai.Client = client
    genai.types = SimpleNamespace(HttpOptions=lambda **kwargs: SimpleNamespace(**kwargs))
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setenv("TEST_GEMINI_KEY", "test-key")

    result = _llm_provider.generate_structured_json(
        _settings(LLMProvider.GEMINI, "TEST_GEMINI_KEY"),
        "system unchanged",
        "user unchanged",
        TEST_SCHEMA,
        "test_schema",
        123,
        7,
    )

    assert result == ["ok"]
    assert captured["client"] == {
        "api_key": "test-key",
        "http_options": SimpleNamespace(timeout=7000),
    }
    request = captured["request"]
    assert isinstance(request, dict)
    assert request["model"] == "provider/test-model"
    assert request["contents"] == "user unchanged"
    assert request["config"] == {
        "system_instruction": "system unchanged",
        "max_output_tokens": 123,
        "response_mime_type": "application/json",
        "response_json_schema": TEST_SCHEMA,
    }


def test_native_gemma_uses_json_mode_with_local_schema_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    google = ModuleType("google")
    genai = ModuleType("google.genai")

    def client(**_kwargs: object) -> SimpleNamespace:
        def generate_content(**request: object) -> SimpleNamespace:
            captured.update(request)
            return SimpleNamespace(text='["ok"]')

        return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

    genai.Client = client
    genai.types = SimpleNamespace(HttpOptions=lambda **kwargs: SimpleNamespace(**kwargs))
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setenv("TEST_GEMINI_KEY", "test-key")
    settings = LLMSettings(
        provider=LLMProvider.GEMINI,
        model="gemma-4-31b-it",
        api_key_environment="TEST_GEMINI_KEY",
    )

    result = _llm_provider.generate_structured_json(
        settings,
        "system",
        "user",
        TEST_SCHEMA,
        "test_schema",
        123,
        7,
    )

    assert result == ["ok"]
    config = captured["config"]
    assert isinstance(config, dict)
    assert config["response_mime_type"] == "application/json"
    assert "response_json_schema" not in config


def test_local_schema_validation_rejects_malformed_provider_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "test-key")
    monkeypatch.setattr(
        _llm_provider,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse({
            "choices": [{"message": {"content": '["ok", 3]'}}],
        }),
    )

    with pytest.raises(ProviderRequestError, match="does not match the response schema") as caught:
        _llm_provider.generate_structured_json(
            _settings(LLMProvider.OPENROUTER, "TEST_OPENROUTER_KEY"),
            "system",
            "user",
            TEST_SCHEMA,
            "test_schema",
            123,
            7,
        )

    assert provider_error_is_retryable(caught.value) is False


@pytest.mark.parametrize("content", ['[NaN]', '[Infinity]', '[1e400]'])
def test_local_schema_validation_rejects_non_finite_numbers(
    monkeypatch: pytest.MonkeyPatch,
    content: str,
) -> None:
    number_schema: dict[str, object] = {
        "type": "array",
        "items": {"type": "number"},
    }
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "test-key")
    monkeypatch.setattr(
        _llm_provider,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse({
            "choices": [{"message": {"content": content}}],
        }),
    )

    with pytest.raises(ProviderRequestError, match="does not match the response schema") as caught:
        _llm_provider.generate_structured_json(
            _settings(LLMProvider.OPENROUTER, "TEST_OPENROUTER_KEY"),
            "system",
            "user",
            number_schema,
            "test_schema",
            123,
            7,
        )

    assert provider_error_is_retryable(caught.value) is False


def test_openrouter_uses_chat_messages_and_strict_json_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, *, timeout: int) -> FakeResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse({
            "choices": [{"message": {"content": '["ok"]'}}],
        })

    monkeypatch.setattr(_llm_provider, "urlopen", fake_urlopen)
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "test-key")

    result = _llm_provider.generate_structured_json(
        _settings(LLMProvider.OPENROUTER, "TEST_OPENROUTER_KEY"),
        "system unchanged",
        "user unchanged",
        TEST_SCHEMA,
        "test_schema",
        123,
        7,
    )

    assert result == ["ok"]
    assert captured["timeout"] == 7
    request = captured["request"]
    assert isinstance(request, _llm_provider.Request)
    payload = json.loads(request.data.decode("utf-8"))
    assert payload == {
        "model": "provider/test-model",
        "messages": [
            {"role": "system", "content": "system unchanged"},
            {"role": "user", "content": "user unchanged"},
        ],
        "max_tokens": 123,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "strict": True,
                "schema": TEST_SCHEMA,
            },
        },
        "provider": {"require_parameters": True},
    }
    assert dict(request.header_items())["Authorization"] == "Bearer test-key"


@pytest.mark.parametrize(
    ("status_code", "retryable"),
    [(401, False), (402, False), (404, False), (429, True), (503, True)],
)
def test_openrouter_http_errors_have_provider_aware_retry_rules(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    retryable: bool,
) -> None:
    def fail_request(*_args: object, **_kwargs: object) -> None:
        body = BytesIO(b'{"error":{"message":"leaked-test-key"}}')
        raise HTTPError(_llm_provider.OPENROUTER_CHAT_COMPLETIONS_URL, status_code, "failure", {}, body)

    monkeypatch.setattr(_llm_provider, "urlopen", fail_request)
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "test-key")

    with pytest.raises(ProviderRequestError) as caught:
        _llm_provider.generate_structured_json(
            _settings(LLMProvider.OPENROUTER, "TEST_OPENROUTER_KEY"),
            "system",
            "user",
            TEST_SCHEMA,
            "test_schema",
            123,
            7,
        )

    assert caught.value.provider == "openrouter"
    assert caught.value.status_code == status_code
    assert provider_error_is_retryable(caught.value) is retryable
    assert "test-key" not in str(caught.value)


def test_missing_key_environment_fails_without_retry_or_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_PROVIDER_KEY", raising=False)
    settings = _settings(LLMProvider.OPENROUTER, "MISSING_PROVIDER_KEY")

    with pytest.raises(ProviderRequestError, match="MISSING_PROVIDER_KEY") as caught:
        _llm_provider.generate_structured_json(
            settings,
            "system",
            "user",
            TEST_SCHEMA,
            "test_schema",
            123,
            7,
        )

    assert provider_error_is_retryable(caught.value) is False
    assert settings.provenance() == {
        "provider": "openrouter",
        "model": "provider/test-model",
        "key_environment": "MISSING_PROVIDER_KEY",
    }
