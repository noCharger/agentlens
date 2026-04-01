from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from agentlens.config import AgentLensSettings
from agentlens.model_selection import resolve_model_selection


class DeepSeekPreflightError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DeepSeekBalanceSummary:
    is_available: bool
    balances: tuple[dict[str, str], ...]

    @property
    def formatted_totals(self) -> str:
        if not self.balances:
            return "no balance details returned"

        parts: list[str] = []
        for balance in self.balances:
            currency = str(balance.get("currency", "UNKNOWN"))
            total = str(balance.get("total_balance", "0"))
            parts.append(f"{currency} {total}")
        return ", ".join(parts)


def _normalize_deepseek_api_base(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized or "https://api.deepseek.com"


def _parse_balance_summary(payload: dict[str, Any]) -> DeepSeekBalanceSummary:
    raw_balances = payload.get("balance_infos")
    balances: list[dict[str, str]] = []

    if isinstance(raw_balances, list):
        for item in raw_balances:
            if not isinstance(item, dict):
                continue
            balances.append(
                {
                    "currency": str(item.get("currency", "")),
                    "total_balance": str(item.get("total_balance", "0")),
                    "granted_balance": str(item.get("granted_balance", "0")),
                    "topped_up_balance": str(item.get("topped_up_balance", "0")),
                }
            )

    is_available = bool(payload.get("is_available"))
    if not is_available and balances:
        for balance in balances:
            try:
                total = Decimal(balance.get("total_balance", "0"))
            except (InvalidOperation, TypeError):
                continue
            if total > 0:
                is_available = True
                break

    return DeepSeekBalanceSummary(
        is_available=is_available,
        balances=tuple(balances),
    )


def fetch_deepseek_balance(
    *,
    api_key: str,
    base_url: str,
    timeout_s: float = 10.0,
) -> DeepSeekBalanceSummary:
    endpoint = f"{_normalize_deepseek_api_base(base_url)}/user/balance"
    response = httpx.get(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout_s,
    )

    if response.status_code == 401:
        raise DeepSeekPreflightError(
            "DeepSeek authentication failed (401). Check DEEPSEEK_API_KEY."
        )
    if response.status_code == 402:
        raise DeepSeekPreflightError(
            "DeepSeek account has insufficient balance (402). "
            "Top up the API account before running evals."
        )
    if response.status_code >= 400:
        detail = ""
        try:
            payload = response.json()
            detail = str(payload.get("error") or payload)
        except Exception:
            detail = response.text.strip()
        raise DeepSeekPreflightError(
            f"DeepSeek balance check failed ({response.status_code}). {detail}".strip()
        )

    return _parse_balance_summary(response.json())


def validate_deepseek_preflight(
    settings: AgentLensSettings,
    *,
    require_judge: bool = False,
) -> DeepSeekBalanceSummary | None:
    selections = [resolve_model_selection(settings.agent_model)]
    if require_judge:
        selections.append(resolve_model_selection(settings.judge_model))

    if not any(selection.provider == "deepseek" for selection in selections):
        return None

    if not settings.deepseek_api_key:
        raise DeepSeekPreflightError(
            "DEEPSEEK_API_KEY is required when using a DeepSeek model."
        )

    summary = fetch_deepseek_balance(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_api_base,
    )
    if not summary.is_available:
        raise DeepSeekPreflightError(
            "DeepSeek balance check succeeded but the account is not available for API calls. "
            f"Reported totals: {summary.formatted_totals}."
        )
    return summary
