from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

from agentlens.config import AgentLensSettings


def create_tracer_provider(
    settings: AgentLensSettings,
    extra_exporters: list[SpanExporter] | None = None,
) -> TracerProvider:
    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)

    otlp_exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    for exporter in extra_exporters or []:
        provider.add_span_processor(SimpleSpanProcessor(exporter))

    return provider


def create_meter_provider(settings: AgentLensSettings) -> MeterProvider:
    resource = Resource.create({"service.name": settings.otel_service_name})
    otlp_metric_exporter = OTLPMetricExporter(endpoint=settings.otel_exporter_otlp_endpoint)
    reader = PeriodicExportingMetricReader(otlp_metric_exporter, export_interval_millis=10000)
    return MeterProvider(resource=resource, metric_readers=[reader])


def init_telemetry(
    settings: AgentLensSettings,
    extra_exporters: list[SpanExporter] | None = None,
) -> tuple[TracerProvider, MeterProvider]:
    tracer_provider = create_tracer_provider(settings, extra_exporters)
    meter_provider = create_meter_provider(settings)

    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(meter_provider)

    return tracer_provider, meter_provider
