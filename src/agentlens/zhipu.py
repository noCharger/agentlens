from __future__ import annotations

from dataclasses import dataclass

from agentlens.config import AgentLensSettings
from agentlens.model_selection import resolve_model_selection


class ZhipuPreflightError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ZhipuPreflightSummary:
    base_url: str

    @property
    def formatted_status(self) -> str:
        return f"base={self.base_url}"


def _normalize_zhipu_api_base(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        return "https://open.bigmodel.cn/api/paas/v4"

    lowered = normalized.casefold()
    if lowered in {"https://open.bigmodel.cn", "http://open.bigmodel.cn"}:
        return f"{normalized}/api/paas/v4"
    if lowered.endswith("/api/paas"):
        return f"{normalized}/v4"
    return normalized


def validate_zhipu_preflight(
    settings: AgentLensSettings,
    *,
    require_judge: bool = False,
) -> ZhipuPreflightSummary | None:
    selections = [resolve_model_selection(settings.agent_model)]
    if require_judge:
        selections.append(resolve_model_selection(settings.judge_model))

    if not any(selection.provider == "zhipu" for selection in selections):
        return None

    if not settings.zhipu_api_key:
        raise ZhipuPreflightError(
            "ZHIPU_API_KEY is required when using a Zhipu model."
        )

    return ZhipuPreflightSummary(
        base_url=_normalize_zhipu_api_base(settings.zhipu_api_base),
    )
