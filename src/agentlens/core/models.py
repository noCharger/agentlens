"""Canonical platform record types for the AgentLens closed loop."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TraceStatus(str, Enum):
    PASSED = "passed"
    PARTIAL_SUCCESS = "partial_success"
    RISKY_SUCCESS = "risky_success"
    FAILED = "failed"
    ERROR = "error"


class DatasetSource(str, Enum):
    TRACE_PROMOTION = "trace_promotion"
    BENCHMARK_IMPORT = "benchmark_import"
    MANUAL_CURATION = "manual_curation"


class AnnotationStatus(str, Enum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    RESOLVED = "resolved"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Role(str, Enum):
    VIEWER = "viewer"
    ENGINEER = "engineer"
    QA = "qa"
    PM = "pm"
    BUSINESS = "business"
    ADMIN = "admin"
    COMPLIANCE = "compliance"


class ProjectRecord(BaseModel):
    id: str
    name: str
    slug: str
    created_at: datetime = Field(default_factory=utc_now)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


class TraceRecord(BaseModel):
    id: str
    scenario_id: str
    scenario_name: str
    benchmark: str = ""
    category: str
    evaluation_mode: str
    status: TraceStatus
    created_at: datetime = Field(default_factory=utc_now)
    input_query: str
    output_text: str = ""
    expected_tools: list[str] = Field(default_factory=list)
    actual_tools: list[str] = Field(default_factory=list)
    missing_tools: list[str] = Field(default_factory=list)
    expected_output_contains: list[str] = Field(default_factory=list)
    missing_output_contains: list[str] = Field(default_factory=list)
    total_steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    judge_scores: dict[str, float] = Field(default_factory=dict)
    judge_overall_score: float | None = None
    error: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class DatasetItemRecord(BaseModel):
    id: str
    dataset_version_id: str
    source_trace_id: str | None = None
    scenario_id: str
    name: str
    category: str
    benchmark: str = ""
    evaluation_mode: str
    input_query: str
    setup_commands: list[str] = Field(default_factory=list)
    max_steps: int = 10
    max_tokens: int | None = None
    reference_answer: str = ""
    expected_tools: list[str] = Field(default_factory=list)
    expected_output_contains: list[str] = Field(default_factory=list)
    judge_rubric: str = ""
    judge_rubric_text: str = ""
    judge_threshold: float = 4.0
    metadata: dict[str, object] = Field(default_factory=dict)


class DatasetVersionRecord(BaseModel):
    id: str
    dataset_id: str
    name: str
    version: str
    source: DatasetSource
    created_at: datetime = Field(default_factory=utc_now)
    items: list[DatasetItemRecord] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def item_count(self) -> int:
        return len(self.items)


class EvalCaseRecord(BaseModel):
    id: str
    trace_id: str | None = None
    scenario_id: str
    scenario_name: str
    benchmark: str = ""
    category: str
    evaluation_mode: str
    passed: bool
    status: TraceStatus = TraceStatus.FAILED
    level1_passed: bool
    judge_overall_score: float | None = None
    risk_signals: list[str] = Field(default_factory=list)
    error: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class EvalRunSummary(BaseModel):
    total: int
    passed: int
    partial_success: int = 0
    risky_success: int = 0
    failed: int
    pass_rate: float
    benchmarks: list[str] = Field(default_factory=list)


class EvalRunRecord(BaseModel):
    id: str
    name: str
    source: str
    created_at: datetime = Field(default_factory=utc_now)
    dataset_version_id: str | None = None
    agent_model: str = ""
    judge_model: str = ""
    cases: list[EvalCaseRecord] = Field(default_factory=list)
    summary: EvalRunSummary
    metadata: dict[str, object] = Field(default_factory=dict)


class ExperimentRecord(BaseModel):
    id: str
    name: str
    created_at: datetime = Field(default_factory=utc_now)
    baseline_run_id: str
    candidate_run_id: str
    baseline_pass_rate: float
    candidate_pass_rate: float
    delta_pass_rate: float
    metadata: dict[str, object] = Field(default_factory=dict)


class AnnotationTaskRecord(BaseModel):
    id: str
    trace_id: str | None = None
    dataset_item_id: str | None = None
    title: str
    status: AnnotationStatus = AnnotationStatus.PENDING
    priority: int = 2
    requested_role: Role = Role.QA
    reason: str
    metadata: dict[str, object] = Field(default_factory=dict)


class AlertRuleRecord(BaseModel):
    id: str
    name: str
    metric_key: str
    operator: str
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    metadata: dict[str, object] = Field(default_factory=dict)


class AlertEventRecord(BaseModel):
    id: str
    rule_id: str
    eval_run_id: str
    metric_key: str
    operator: str
    threshold: float
    observed_value: float
    severity: AlertSeverity
    message: str
    triggered_at: datetime = Field(default_factory=utc_now)
    acknowledged: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)


class AuditEventRecord(BaseModel):
    id: str
    action: str
    actor_role: Role
    resource_type: str
    resource_id: str
    occurred_at: datetime = Field(default_factory=utc_now)
    details: dict[str, object] = Field(default_factory=dict)
