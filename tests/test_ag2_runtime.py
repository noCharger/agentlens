import json
from types import SimpleNamespace

from opentelemetry.sdk.trace import TracerProvider

from agentlens.agents.runtime import AG2Runtime
from agentlens.agents.runtime import _build_ag2_llm_config, _zhipu_price_for_model
from agentlens.eval.level1_deterministic.output_format import extract_output
from agentlens.eval.level1_deterministic.tool_params import extract_tool_params
from agentlens.eval.level1_deterministic.tool_usage import extract_tool_names
from agentlens.eval.level1_deterministic.trajectory import count_steps, sum_tokens
from agentlens.eval.runner import EvalResult, Level1Result, _record_metrics_best_effort
from agentlens.eval.level1_deterministic.output_format import OutputFormatResult
from agentlens.eval.level1_deterministic.tool_usage import ToolUsageResult
from agentlens.eval.level1_deterministic.trajectory import TrajectoryResult
from agentlens.eval.scenarios import ExpectedResult, Scenario
from agentlens.observability.instrument import instrument_runtime


class _FakeStatusCode:
    name = "OK"


class _FakeStatus:
    status_code = _FakeStatusCode()


class _FakeSpan:
    def __init__(self, name, attributes, start_time=1, end_time=2, status=None):
        self.name = name
        self.attributes = attributes
        self.start_time = start_time
        self.end_time = end_time
        self.status = status or _FakeStatus()


def _settings():
    return SimpleNamespace(
        agent_framework="ag2",
        agent_model="openrouter:openai/gpt-4o-mini",
        agent_max_tokens=128,
        openrouter_api_key="test-key",
        openrouter_api_base="https://openrouter.ai/api/v1",
        openrouter_http_referer=None,
        openrouter_x_title=None,
        google_api_key=None,
        deepseek_api_key=None,
        deepseek_api_base="https://api.deepseek.com",
        zhipu_api_key=None,
        zhipu_api_base="https://open.bigmodel.cn/api/paas/v4",
    )


def test_ag2_runtime_builds_public_tool_names():
    runtime = AG2Runtime(_settings(), preset="shell_file")

    assert [tool.name for tool in runtime.tools] == ["terminal", "read_file", "write_file"]


def test_ag2_runtime_normalizes_spans_for_existing_evaluators():
    runtime = AG2Runtime(_settings(), preset="shell_file")
    spans = [
        _FakeSpan(
            "invoke_agent assistant",
            {
                "ag2.span.type": "agent",
                "gen_ai.agent.name": "assistant",
                "gen_ai.output.messages": json.dumps([
                    {"role": "assistant", "parts": [{"type": "text", "content": "hello world TERMINATE"}]}
                ]),
            },
            start_time=10,
            end_time=20,
        ),
        _FakeSpan(
            "chat openai/gpt-4o-mini",
            {
                "ag2.span.type": "llm",
                "gen_ai.request.model": "openai/gpt-4o-mini",
                "gen_ai.usage.input_tokens": 12,
                "gen_ai.usage.output_tokens": 5,
            },
            start_time=11,
            end_time=19,
        ),
        _FakeSpan(
            "execute_tool terminal",
            {
                "ag2.span.type": "tool",
                "gen_ai.tool.name": "terminal",
                "gen_ai.tool.call.arguments": '{"commands":"ls"}',
                "gen_ai.tool.call.result": "ok",
            },
            start_time=12,
            end_time=18,
        ),
    ]

    normalized = runtime.normalize_spans(spans)

    assert extract_tool_names(normalized) == ["terminal"]
    assert extract_output(normalized) == "hello world"
    assert count_steps(normalized) == 1
    assert extract_tool_params(normalized)[0]["params_raw"] == '{"commands":"ls"}'
    assert sum_tokens(normalized) == (12, 5)


def test_record_metrics_best_effort_consumes_normalized_ag2_spans(monkeypatch):
    recorded = {"tool_calls": [], "llm_calls": []}

    class FakeMetrics:
        def record_agent_run(self, **kwargs):
            return None

        def record_eval_outcome(self, *args, **kwargs):
            return None

        def record_risk_signal_count(self, *args, **kwargs):
            return None

        def record_risk_signal(self, *args, **kwargs):
            return None

        def record_failure_pattern(self, *args, **kwargs):
            return None

        def record_judge_score(self, *args, **kwargs):
            return None

        def record_tool_call(self, **kwargs):
            recorded["tool_calls"].append(kwargs)

        def record_llm_call(self, **kwargs):
            recorded["llm_calls"].append(kwargs)

    monkeypatch.setattr("agentlens.observability.metrics.AgentMetrics", FakeMetrics)

    runtime = AG2Runtime(_settings(), preset="shell")
    normalized = runtime.normalize_spans([
        _FakeSpan(
            "chat openai/gpt-4o-mini",
            {
                "ag2.span.type": "llm",
                "gen_ai.request.model": "openai/gpt-4o-mini",
                "gen_ai.usage.input_tokens": 7,
                "gen_ai.usage.output_tokens": 3,
            },
            start_time=10,
            end_time=20,
        ),
        _FakeSpan(
            "execute_tool terminal",
            {
                "ag2.span.type": "tool",
                "gen_ai.tool.name": "terminal",
                "gen_ai.tool.call.arguments": '{"commands":"pwd"}',
                "gen_ai.tool.call.result": "/tmp",
            },
            start_time=12,
            end_time=18,
        ),
    ])

    scenario = Scenario(
        id="ag2-001",
        name="AG2",
        category="tool_calling",
        input="query",
        setup=[],
        expected=ExpectedResult(tools_called=["terminal"], max_steps=4, output_contains=[]),
    )
    result = EvalResult(
        scenario=scenario,
        level1=Level1Result(
            tool_usage=ToolUsageResult(True, ["terminal"], ["terminal"], [], []),
            output_format=OutputFormatResult(True, "ok", [], []),
            trajectory=TrajectoryResult(True, 1, 4, False, 7, 3, None, []),
        ),
    )

    _record_metrics_best_effort(normalized, scenario, result)

    assert recorded["tool_calls"][0]["tool_name"] == "terminal"
    assert recorded["llm_calls"][0]["prompt_tokens"] == 7
    assert recorded["llm_calls"][0]["completion_tokens"] == 3


def test_ag2_runtime_invoke_uses_executor_termination_override(monkeypatch):
    runtime = AG2Runtime(_settings(), preset="shell")
    captured = {}

    class FakeResponse:
        events = []
        summary = "final answer"
        messages = []

    def fake_run(**kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(runtime.agent, "run", fake_run)

    result = runtime.invoke("query", max_steps=3)

    assert result.output_text == "final answer"
    terminate = captured["executor_kwargs"]["is_termination_msg"]
    assert terminate({"role": "assistant", "content": "done"})
    assert terminate({"role": "user", "name": "assistant", "content": "done"})
    assert not terminate({"role": "assistant", "content": None})
    assert not terminate({"role": "assistant", "content": "", "tool_calls": []})
    assert not terminate({
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_1", "function": {"name": "terminal", "arguments": "{}"}}],
    })


def test_ag2_instrumentor_restores_openaiwrapper_create(monkeypatch):
    from autogen.oai import client as oai_client_module
    from autogen.oai.client import OpenAIWrapper

    original_create = OpenAIWrapper.create

    def fake_instrument_llm_wrapper(*, tracer_provider, capture_messages=False):
        def traced_create(self, **config):
            return {"config": config}

        traced_create.__otel_wrapped__ = True
        OpenAIWrapper.create = traced_create
        oai_client_module.OpenAIWrapper.create = traced_create

    monkeypatch.setattr(
        "autogen.opentelemetry.instrument_llm_wrapper",
        fake_instrument_llm_wrapper,
    )
    monkeypatch.setattr("autogen.opentelemetry.instrument_agent", lambda target, tracer_provider: None)

    instrumentor = instrument_runtime("ag2", TracerProvider(), target=SimpleNamespace())

    assert OpenAIWrapper.create is not original_create

    instrumentor.uninstrument()

    assert OpenAIWrapper.create is original_create


def test_zhipu_price_for_model_known_aliases():
    assert _zhipu_price_for_model("glm-4-flash") == (0.0, 0.0)
    assert _zhipu_price_for_model("glm-4-flashx") == (0.0001, 0.0001)
    assert _zhipu_price_for_model("glm-4-air") == (0.0005, 0.0005)
    assert _zhipu_price_for_model("glm-4-plus") == (0.005, 0.005)
    assert _zhipu_price_for_model("unknown-zhipu-model") is None


def test_build_ag2_llm_config_includes_zhipu_price():
    settings = SimpleNamespace(
        agent_framework="ag2",
        agent_model="zhipu:glm-4-flash",
        agent_max_tokens=256,
        openrouter_api_key=None,
        openrouter_api_base="https://openrouter.ai/api/v1",
        openrouter_http_referer=None,
        openrouter_x_title=None,
        google_api_key=None,
        deepseek_api_key=None,
        deepseek_api_base="https://api.deepseek.com",
        zhipu_api_key="zp-test",
        zhipu_api_base="https://open.bigmodel.cn/api/paas/v4",
    )

    llm_config = _build_ag2_llm_config(settings)
    config = llm_config.model_dump()["config_list"][0]

    assert config["model"] == "glm-4-flash"
    assert config["price"] == [0.0, 0.0]
