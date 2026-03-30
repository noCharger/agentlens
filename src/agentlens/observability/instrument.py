from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.langchain import LangChainInstrumentor


def instrument_langchain(tracer_provider: TracerProvider) -> LangChainInstrumentor:
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    return instrumentor


def uninstrument_langchain(instrumentor: LangChainInstrumentor) -> None:
    instrumentor.uninstrument()
