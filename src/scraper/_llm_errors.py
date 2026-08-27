"""Represent and classify LLM failures that cannot recover during a run."""

_DAILY_REQUEST_QUOTA_BASE_ID = "GenerateRequestsPerDayPerProjectPerModel"


class DailyRequestQuotaError(RuntimeError):
    """Raised when no later LLM request can succeed during this quota day."""


class ProviderRequestError(RuntimeError):
    """A safe provider-boundary failure with an explicit retry decision."""

    def __init__(self, provider: str, message: str, *, retryable: bool, status_code: int | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
        self.status_code = status_code


def provider_error_is_retryable(error: Exception) -> bool:
    """Return whether a provider-boundary failure can recover on retry.

    Native Gemini SDK errors retain the existing broad retry behaviour. The
    provider boundary classifies OpenRouter and local configuration failures.

    :param error: request or response failure.
    :return: explicit provider decision, or True for existing unclassified errors.
    """
    if isinstance(error, ProviderRequestError):
        return error.retryable
    return True


def daily_request_quota_exhausted(error: Exception) -> bool:
    """Return whether Google reported exhaustion of the model's daily requests.

    The optional Google SDK is not installed in every supported environment,
    so this boundary reads its stable ``code`` and structured ``details``
    attributes without importing provider exception classes.

    :param error: provider or transport exception raised by an LLM request.
    :return: whether retrying during the current quota day cannot succeed.
    """
    if getattr(error, "code", None) != 429:
        return False
    return _contains_daily_request_quota(getattr(error, "details", None))


def _contains_daily_request_quota(payload: object) -> bool:
    if isinstance(payload, dict):
        quota_id = payload.get("quotaId")
        if isinstance(quota_id, str) and (
            quota_id == _DAILY_REQUEST_QUOTA_BASE_ID
            or quota_id.startswith(f"{_DAILY_REQUEST_QUOTA_BASE_ID}-")
        ):
            return True
        for value in payload.values():
            if _contains_daily_request_quota(value):
                return True
    elif isinstance(payload, list):
        for value in payload:
            if _contains_daily_request_quota(value):
                return True
    return False
