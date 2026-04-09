from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.langchain import LangChainInstrumentor


class Instrumentor(Protocol):
    def uninstrument(self) -> None: ...


@dataclass
class NoopInstrumentor:
    def uninstrument(self) -> None:
        return None


@dataclass
class AG2Instrumentor:
    agent: Any
    original_openaiwrapper_create: Any | None = None

    def uninstrument(self) -> None:
        if self.original_openaiwrapper_create is None:
            return None
        try:
            from autogen.oai import client as oai_client_module
            from autogen.oai.client import OpenAIWrapper
        except ImportError:
            return None

        OpenAIWrapper.create = self.original_openaiwrapper_create
        oai_client_module.OpenAIWrapper.create = self.original_openaiwrapper_create
        return None


def instrument_runtime(
    framework: str,
    tracer_provider: TracerProvider,
    *,
    target: Any | None = None,
) -> Instrumentor:
    if framework == "langgraph":
        instrumentor = LangChainInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)
        return instrumentor

    if framework == "ag2":
        if target is None:
            raise ValueError("AG2 instrumentation requires an agent target.")
        try:
            from autogen.opentelemetry import instrument_agent, instrument_llm_wrapper
            from autogen.oai.client import OpenAIWrapper
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "AG2 tracing requires the 'ag2' package. Install project dependencies again after "
                "updating pyproject.toml."
            ) from exc
        original_openaiwrapper_create = OpenAIWrapper.create
        instrument_llm_wrapper(tracer_provider=tracer_provider)
        instrument_agent(target, tracer_provider=tracer_provider)
        return AG2Instrumentor(
            agent=target,
            original_openaiwrapper_create=original_openaiwrapper_create,
        )

    raise ValueError(f"Unsupported agent framework '{framework}'.")


def uninstrument_runtime(instrumentor: Instrumentor) -> None:
    instrumentor.uninstrument()


# Backward-compatible aliases used by older tests/import sites.
def instrument_langchain(tracer_provider: TracerProvider) -> LangChainInstrumentor:
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    return instrumentor


def uninstrument_langchain(instrumentor: LangChainInstrumentor) -> None:
    instrumentor.uninstrument()
