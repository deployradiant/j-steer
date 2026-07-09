from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RunMode = Literal["monitor", "rewrite"]
MonitorAction = Literal["observe", "kill"]
RunStatus = Literal["completed", "blocked", "rewritten", "error"]
FrameAction = Literal["observe", "blocked", "rewrite"]
ConceptPolarity = Literal["risk", "target", "neutral"]


class ConceptScore(BaseModel):
    id: str
    label: str
    score: float = Field(ge=0.0, le=1.0)
    polarity: ConceptPolarity = "neutral"
    evidence: list[str] = Field(default_factory=list)


class TraceFrame(BaseModel):
    step: int = Field(ge=0)
    phase: Literal["prompt", "decode"]
    layer: int = Field(ge=0)
    token: str = ""
    generated_text: str = ""
    top_concepts: list[ConceptScore] = Field(default_factory=list)
    action: FrameAction = "observe"
    note: str | None = None


class Intervention(BaseModel):
    step: int = Field(ge=0)
    from_concept: str
    to_concept: str
    strength: float
    reason: str


class EngineInfo(BaseModel):
    mode: str
    model: str
    layer: int
    device: str = "modal"
    readout: str


class RunMetrics(BaseModel):
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    peak_score: float = 0.0
    rewrites: int = 0


class RunRequest(BaseModel):
    mode: RunMode = "monitor"
    prompt: str = Field(min_length=1, max_length=8000)
    monitor_ids: list[str] = Field(default_factory=list)
    rewrite_profile_id: str | None = None
    action: MonitorAction = "kill"
    threshold: float = Field(default=0.66, ge=0.05, le=0.99)
    max_new_tokens: int = Field(default=90, ge=8, le=256)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.05, le=1.0)
    strength: float = Field(default=1.15, ge=0.0, le=4.0)
    compare_baseline: bool = True
    seed: int = Field(default=7, ge=0)


class RunResponse(BaseModel):
    id: str
    status: RunStatus
    mode: RunMode
    prompt: str
    output: str
    baseline_output: str | None = None
    summary: str
    engine: EngineInfo
    frames: list[TraceFrame] = Field(default_factory=list)
    baseline_frames: list[TraceFrame] = Field(default_factory=list)
    interventions: list[Intervention] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)


class ConceptPublic(BaseModel):
    id: str
    label: str
    description: str
    polarity: ConceptPolarity
    keywords: list[str]


class RewriteProfilePublic(BaseModel):
    id: str
    label: str
    description: str
    target_label: str
    suppress_labels: list[str]
    default_strength: float
    demo_prompt: str


class ScenarioPublic(BaseModel):
    id: str
    mode: RunMode
    label: str
    prompt: str
    monitor_ids: list[str] = Field(default_factory=list)
    rewrite_profile_id: str | None = None
    action: MonitorAction = "kill"
    threshold: float = 0.66


class ProfilesResponse(BaseModel):
    concepts: list[ConceptPublic]
    rewrite_profiles: list[RewriteProfilePublic]
    scenarios: list[ScenarioPublic]
