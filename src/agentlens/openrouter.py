from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from agentlens.config import AgentLensSettings
from agentlens.model_selection import resolve_model_selection


class OpenRouterPreflightError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OpenRouterKeySummary:
    is_available: bool
    label: str | None
    usage: str | None
    limit: str | None
    limit_remaining: str | None
    is_free_tier: bool | None
    is_provisioning_key: bool | None

    @property
    def formatted_status(self) -> str:
        parts: list[str] = []
        if self.label:
            parts.append(f"label={self.label}")
        if self.limit_remaining is not None:
            if self.limit is not None:
                parts.append(f"remaining {self.limit_remaining}/{self.limit}")
            else:
                parts.append(f"remaining {self.limit_remaining}")
        if self.usage is not None:
            parts.append(f"usage {self.usage}")
        if self.is_free_tier is not None:
            parts.append("free-tier" if self.is_free_tier else "paid-tier")
        if self.is_provisioning_key is not None:
            parts.append(
                "provisioning-key" if self.is_provisioning_key else "inference-key"
            )
        return ", ".join(parts) if parts else "key metadata available"


def _normalize_openrouter_api_base(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        return "https://openrouter.ai/api/v1"

    lowered = normalized.casefold()
    if lowered in {"https://openrouter.ai", "http://openrouter.ai"}:
        return f"{normalized}/api/v1"
    if lowered.endswith("/api"):
        return f"{normalized}/v1"
    return normalized


def _parse_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _parse_key_summary(payload: dict[str, Any]) -> OpenRouterKeySummary:
    data = payload.get("data")
    if not isinstance(data, dict):
        data = payload

    limit_remaining = _parse_decimal(data.get("limit_remaining"))
    disabled = bool(data.get("disabled"))
    is_available = not disabled
    if limit_remaining is not None and limit_remaining <= 0:
        is_available = False

    return OpenRouterKeySummary(
        is_available=is_available,
        label=str(data.get("label")) if data.get("label") is not None else None,
        usage=str(data.get("usage")) if data.get("usage") is not None else None,
        limit=str(data.get("limit")) if data.get("limit") is not None else None,
        limit_remaining=(
            str(data.get("limit_remaining"))
            if data.get("limit_remaining") is not None
            else None
        ),
        is_free_tier=(
            bool(data.get("is_free_tier"))
            if data.get("is_free_tier") is not None
            else None
        ),
        is_provisioning_key=(
            bool(data.get("is_provisioning_key"))
            if data.get("is_provisioning_key") is not None
            else None
        ),
    )


def fetch_openrouter_key_info(
    *,
    api_key: str,
    base_url: str,
    timeout_s: float = 10.0,
) -> OpenRouterKeySummary:
    endpoint = f"{_normalize_openrouter_api_base(base_url)}/key"
    try:
        response = httpx.get(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
        )
    except httpx.HTTPError as exc:
        raise OpenRouterPreflightError(
            f"OpenRouter key check failed due to network error: {exc}"
        ) from exc

    if response.status_code in {401, 403}:
        raise OpenRouterPreflightError(
            "OpenRouter authentication failed (401/403). Check OPENROUTER_API_KEY."
        )
    if response.status_code == 402:
        raise OpenRouterPreflightError(
            "OpenRouter account has insufficient credits (402). "
            "Top up the API account before running evals."
        )
    if response.status_code >= 400:
        detail = ""
        try:
            payload = response.json()
            if isinstance(payload, dict):
                error = payload.get("error")
                if isinstance(error, dict):
                    detail = str(error.get("message") or error)
                else:
                    detail = str(error or payload)
            else:
                detail = str(payload)
        except Exception:
            detail = response.text.strip()
        raise OpenRouterPreflightError(
            f"OpenRouter key check failed ({response.status_code}). {detail}".strip()
        )

    return _parse_key_summary(response.json())


def validate_openrouter_preflight(
    settings: AgentLensSettings,
    *,
    require_judge: bool = False,
) -> OpenRouterKeySummary | None:
    selections = [resolve_model_selection(settings.agent_model)]
    if require_judge:
        selections.append(resolve_model_selection(settings.judge_model))

    if not any(selection.provider == "openrouter" for selection in selections):
        return None

    if not settings.openrouter_api_key:
        raise OpenRouterPreflightError(
            "OPENROUTER_API_KEY is required when using an OpenRouter model."
        )

    summary = fetch_openrouter_key_info(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_api_base,
    )
    if not summary.is_available:
        raise OpenRouterPreflightError(
            "OpenRouter key check succeeded but the key is not available for API calls. "
            f"Reported: {summary.formatted_status}."
        )
    return summary
