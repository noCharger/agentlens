from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.metrics import MeterProvider

from agentlens.config import AgentLensSettings
from agentlens.observability.setup import create_tracer_provider, create_meter_provider


def _settings():
    return AgentLensSettings(
        google_api_key="test",
        otel_exporter_otlp_endpoint="http://localhost:4317",
        otel_service_name="test-service",
    )


def test_create_tracer_provider_returns_provider():
    provider = create_tracer_provider(_settings())
    assert isinstance(provider, TracerProvider)
    provider.shutdown()


def test_create_tracer_provider_with_extra_exporter():
    mem_exporter = InMemorySpanExporter()
    provider = create_tracer_provider(_settings(), extra_exporters=[mem_exporter])
    assert isinstance(provider, TracerProvider)

    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("test-span"):
        pass
    provider.force_flush()

    spans = mem_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test-span"
    provider.shutdown()


def test_create_tracer_provider_has_service_name():
    provider = create_tracer_provider(_settings())
    resource = provider.resource
    attrs = dict(resource.attributes)
    assert attrs["service.name"] == "test-service"
    provider.shutdown()


def test_create_meter_provider_returns_provider():
    provider = create_meter_provider(_settings())
    assert isinstance(provider, MeterProvider)
    provider.shutdown()
