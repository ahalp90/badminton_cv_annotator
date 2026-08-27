"""Small provider boundary for structured commentary LLM calls."""

from dataclasses import dataclass
from enum import StrEnum
import json
import math
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ._llm_errors import ProviderRequestError

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
_ENVIRONMENT_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class LLMProvider(StrEnum):
    """Supported commentary LLM providers."""

    GEMINI = "gemini"
    OPENROUTER = "openrouter"


@dataclass(frozen=True)
class LLMSettings:
    """Effective provider settings without a credential value."""

    provider: LLMProvider
    model: str
    api_key_environment: str

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("LLM model must not be blank")
        if not self.api_key_environment.strip():
            raise ValueError("LLM API key environment must not be blank")
        validate_api_key_environment(self.api_key_environment)

    @classmethod
    def from_values(cls, provider: str, model: str, api_key_environment: str) -> "LLMSettings":
        """Validate string configuration at a CLI or TOML boundary."""
        return cls(
            provider=LLMProvider(provider),
            model=model,
            api_key_environment=api_key_environment,
        )

    def provenance(self) -> dict[str, str]:
        """Return safe reproducibility fields without reading the credential."""
        return {
            "provider": self.provider.value,
            "model": self.model,
            "key_environment": self.api_key_environment,
        }


def resolve_cli_settings(
    provider: str,
    model: str | None,
    api_key_environment: str | None,
    *,
    gemini_model: str,
    gemini_api_key_environment: str,
) -> LLMSettings:
    """Apply defaults only to Gemini and require explicit OpenRouter routing."""
    selected = LLMProvider(provider)
    if selected is LLMProvider.GEMINI:
        return LLMSettings(
            provider=selected,
            model=gemini_model if model is None else model,
            api_key_environment=(
                gemini_api_key_environment
                if api_key_environment is None
                else api_key_environment
            ),
        )
    if not model or not api_key_environment:
        raise ValueError(
            "OpenRouter requires explicit --model and --api-key-environment values"
        )
    return LLMSettings(
        provider=selected,
        model=model,
        api_key_environment=api_key_environment,
    )


def validate_api_key_environment(value: str) -> str:
    """Reject values that cannot be portable environment variable names."""
    if _ENVIRONMENT_NAME_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "LLM API key environment must be a portable environment variable name"
        )
    return value


def generate_structured_json(
    settings: LLMSettings,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict[str, object],
    schema_name: str,
    max_output_tokens: int,
    timeout_seconds: int,
) -> object:
    """Generate one JSON value through the selected provider.

    :param settings: provider, model, and credential environment name.
    :param system_prompt: unchanged system instruction from the caller.
    :param user_prompt: unchanged user content from the caller.
    :param response_schema: JSON Schema matching the caller's existing return contract.
    :param schema_name: stable OpenRouter schema name.
    :param max_output_tokens: provider output-token limit.
    :param timeout_seconds: synchronous request timeout.
    :return: parsed JSON response.
    """
    api_key = _api_key(settings)
    if settings.provider is LLMProvider.GEMINI:
        parsed = _generate_gemini(
            settings,
            api_key,
            system_prompt,
            user_prompt,
            response_schema,
            max_output_tokens,
            timeout_seconds,
        )
    else:
        parsed = _generate_openrouter(
            settings,
            api_key,
            system_prompt,
            user_prompt,
            response_schema,
            schema_name,
            max_output_tokens,
            timeout_seconds,
        )
    try:
        _validate_json_value(parsed, response_schema)
    except ValueError as error:
        raise ProviderRequestError(
            settings.provider.value,
            f"{settings.provider.value} returned JSON that does not match the response schema",
            retryable=settings.provider is LLMProvider.GEMINI,
        ) from error
    return parsed


def _api_key(settings: LLMSettings) -> str:
    api_key = os.environ.get(settings.api_key_environment)
    if not api_key:
        message = (
            f"required {settings.provider.value} environment variable is unset: "
            f"{settings.api_key_environment}"
        )
        raise ProviderRequestError(
            settings.provider.value,
            message,
            retryable=False,
        )
    return api_key


def _generate_gemini(
    settings: LLMSettings,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict[str, object],
    max_output_tokens: int,
    timeout_seconds: int,
) -> object:
    # Function-local import keeps the optional SDK out of test and CPU-only environments.
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=timeout_seconds * 1000),
    )
    config: dict[str, object] = {
        "system_instruction": system_prompt,
        "max_output_tokens": max_output_tokens,
        "response_mime_type": "application/json",
    }
    # Native Gemma currently supports JSON mode but is absent from Google's
    # structured-output model list. Local validation below enforces the schema.
    if not settings.model.startswith("gemma-"):
        config["response_json_schema"] = response_schema
    response = client.models.generate_content(
        model=settings.model,
        contents=user_prompt,
        config=config,
    )
    return json.loads(response.text)


def _generate_openrouter(
    settings: LLMSettings,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict[str, object],
    schema_name: str,
    max_output_tokens: int,
    timeout_seconds: int,
) -> object:
    payload = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_output_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": response_schema,
            },
        },
        "provider": {"require_parameters": True},
    }
    request = Request(
        OPENROUTER_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise ProviderRequestError(
            settings.provider.value,
            f"OpenRouter request failed with HTTP {error.code}",
            retryable=_openrouter_status_is_retryable(error.code),
            status_code=error.code,
        ) from error
    except (TimeoutError, URLError, OSError) as error:
        raise ProviderRequestError(
            settings.provider.value,
            f"OpenRouter transport failed: {type(error).__name__}",
            retryable=True,
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProviderRequestError(
            settings.provider.value,
            "OpenRouter returned a malformed JSON envelope",
            retryable=True,
        ) from error

    return _openrouter_content(settings, response_payload)


def _openrouter_content(settings: LLMSettings, response_payload: object) -> object:
    try:
        if not isinstance(response_payload, dict):
            raise TypeError("response envelope is not an object")
        if "error" in response_payload:
            error_payload = response_payload["error"]
            status_code = error_payload.get("code") if isinstance(error_payload, dict) else None
            status_code = status_code if isinstance(status_code, int) else None
            raise ProviderRequestError(
                settings.provider.value,
                "OpenRouter returned an error response",
                retryable=(status_code is None or _openrouter_status_is_retryable(status_code)),
                status_code=status_code,
            )
        choices = response_payload["choices"]
        content = choices[0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("response content is not text")
        return json.loads(content)
    except ProviderRequestError:
        raise
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise ProviderRequestError(
            settings.provider.value,
            "OpenRouter returned a malformed structured response",
            retryable=False,
        ) from error


def _openrouter_status_is_retryable(status_code: int) -> bool:
    return status_code in {408, 429} or status_code >= 500


def _validate_json_value(value: object, schema: dict[str, object], path: str = "$") -> None:
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path} is not an object")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError(f"{path} has an invalid object schema")
        required = schema.get("required", [])
        if not isinstance(required, list) or any(not isinstance(name, str) for name in required):
            raise ValueError(f"{path} has invalid required fields")
        missing = [name for name in required if name not in value]
        if missing:
            raise ValueError(f"{path} is missing required fields")
        if schema.get("additionalProperties") is False and any(name not in properties for name in value):
            raise ValueError(f"{path} has additional fields")
        for name, child_schema in properties.items():
            if name in value:
                if not isinstance(child_schema, dict):
                    raise ValueError(f"{path}.{name} has an invalid schema")
                _validate_json_value(value[name], child_schema, f"{path}.{name}")
        return
    if expected_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path} is not an array")
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if isinstance(minimum, int) and len(value) < minimum:
            raise ValueError(f"{path} has too few items")
        if isinstance(maximum, int) and len(value) > maximum:
            raise ValueError(f"{path} has too many items")
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            raise ValueError(f"{path} has an invalid item schema")
        for index, item in enumerate(value):
            _validate_json_value(item, item_schema, f"{path}[{index}]")
        return
    if expected_type == "string" and not isinstance(value, str):
        raise ValueError(f"{path} is not a string")
    if expected_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{path} is not a number")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{path} is not finite")
