"""AgentProxy: HTTP MITM proxy for non-intrusive LLM API signal capture.

Usage (zero code change in agent):
    export OPENAI_BASE_URL=http://localhost:9999/v1
    agentlens proxy --port 9999

The proxy intercepts OpenAI-compatible request/response pairs and emits
OTEL spans normalized to the same openinference schema used by L1/L2 checks.
No SDK dependency on the agent side — only an env var is required.

Three signal surfaces captured (AgentTrace taxonomy):
- Cognitive:    <thinking> / scratchpad text in response content
- Operational:  tool_calls extracted from choice delta
- Contextual:   token usage, model, latency, message count
"""

from __future__ import annotations

import json
import logging
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError

log = logging.getLogger("agentlens.proxy")

_THINKING_RE = None  # lazy import re pattern


def _extract_thinking(text: str) -> str:
    """Extract <thinking>...</thinking> or similar scratchpad blocks."""
    import re
    pattern = re.compile(r"<(?:thinking|scratchpad)>(.*?)</(?:thinking|scratchpad)>", re.DOTALL)
    matches = pattern.findall(text)
    return "\n".join(m.strip() for m in matches)


def _emit_otel_spans(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_s: float,
    output_text: str,
    tool_calls: list[dict[str, Any]],
    thinking: str,
) -> None:
    """Emit normalized openinference OTEL spans from proxy-captured data."""
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("agentlens.proxy")

        with tracer.start_as_current_span("LLM") as llm_span:
            llm_span.set_attribute("openinference.span.kind", "LLM")
            llm_span.set_attribute("llm.model_name", model)
            llm_span.set_attribute("llm.token_count.prompt", prompt_tokens)
            llm_span.set_attribute("llm.token_count.completion", completion_tokens)
            llm_span.set_attribute("llm.latency_seconds", latency_s)

            if thinking:
                llm_span.set_attribute("step.thought", thinking[:2000])

            if output_text and not tool_calls:
                llm_span.set_attribute("output.value", output_text[:4000])

            for tc in tool_calls:
                fn = tc.get("function", {})
                with tracer.start_as_current_span(f"tool:{fn.get('name', 'unknown')}") as tool_span:
                    tool_span.set_attribute("openinference.span.kind", "TOOL")
                    tool_span.set_attribute("tool.name", fn.get("name", ""))
                    tool_span.set_attribute(
                        "input.value", fn.get("arguments", "")[:2000]
                    )
    except Exception as exc:
        log.debug("Failed to emit proxy OTEL spans: %s", exc)


class _ProxyHandler(BaseHTTPRequestHandler):
    upstream: str = "https://api.openai.com"
    log_message = lambda self, *a, **kw: None  # silence default access log

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}

        upstream_url = self.upstream.rstrip("/") + self.path
        req = Request(
            upstream_url,
            data=body,
            headers={k: v for k, v in self.headers.items()
                     if k.lower() not in ("host", "content-length")},
            method="POST",
        )

        t0 = time.monotonic()
        try:
            with urlopen(req, timeout=120) as resp:
                resp_body = resp.read()
                status = resp.status
                resp_headers = dict(resp.headers)
        except HTTPError as exc:
            resp_body = exc.read()
            status = exc.code
            resp_headers = dict(exc.headers)

        latency = time.monotonic() - t0

        try:
            resp_json = json.loads(resp_body)
            model = resp_json.get("model", payload.get("model", ""))
            usage = resp_json.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            choices = resp_json.get("choices", [])
            output_text = ""
            tool_calls: list[dict] = []
            thinking = ""

            if choices:
                msg = choices[0].get("message") or choices[0].get("delta") or {}
                content = msg.get("content") or ""
                if isinstance(content, list):
                    output_text = " ".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )
                else:
                    output_text = str(content)
                tool_calls = msg.get("tool_calls") or []
                if output_text:
                    thinking = _extract_thinking(output_text)

            _emit_otel_spans(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_s=latency,
                output_text=output_text,
                tool_calls=tool_calls,
                thinking=thinking,
            )
        except Exception as exc:
            log.debug("Proxy signal extraction failed: %s", exc)

        self.send_response(status)
        for k, v in resp_headers.items():
            if k.lower() in ("content-type", "transfer-encoding"):
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"agentlens-proxy-ok"}')


class AgentProxy:
    """Non-intrusive HTTP MITM proxy for any OpenAI-compatible agent."""

    def __init__(self, *, port: int = 9999, upstream: str = "https://api.openai.com"):
        self.port = port
        self.upstream = upstream
        self._server: HTTPServer | None = None
        self._thread: Thread | None = None

    def start(self) -> None:
        handler = type(
            "_BoundHandler",
            (_ProxyHandler,),
            {"upstream": self.upstream},
        )
        self._server = HTTPServer(("localhost", self.port), handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        log.info("AgentProxy listening on http://localhost:%d → %s", self.port, self.upstream)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
        log.info("AgentProxy stopped.")

    def __enter__(self) -> "AgentProxy":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()


def run_proxy_server(port: int = 9999, upstream: str = "https://api.openai.com") -> None:
    """Blocking entrypoint for `agentlens proxy` CLI command."""
    proxy = AgentProxy(port=port, upstream=upstream)
    proxy.start()
    print(f"AgentProxy running on http://localhost:{port}")
    print(f"Set OPENAI_BASE_URL=http://localhost:{port}/v1 in your agent process.")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        proxy.stop()
