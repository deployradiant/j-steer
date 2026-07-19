from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RunMode = Literal["monitor", "rewrite"]
MonitorAction = Literal["observe", "kill"]
RunStatus = Literal["running", "completed", "blocked", "rewritten", "cancelled", "error"]
FrameAction = Literal["observe", "blocked", "rewrite"]
ConceptPolarity = Literal["risk", "target", "neutral"]
InjectionTarget = Literal["custom_lens", "rewrite_controller"]


class ConceptScore(BaseModel):
    id: str
    label: str
    score: float = Field(ge=0.0, le=1.0)
    polarity: ConceptPolarity = "neutral"
    evidence: list[str] = Field(default_factory=list)


class LayerActivation(BaseModel):
    layer: int = Field(ge=0)
    concept_id: str
    label: str
    score: float = Field(ge=0.0, le=1.0)
    polarity: ConceptPolarity = "neutral"
    rank: int | None = Field(default=None, ge=1)
    matched_token: str | None = None
    top_tokens: list[str] = Field(default_factory=list)


class JacobianLensInfo(BaseModel):
    id: str = ""
    label: str = ""
    available: bool = False
    status: str = "configured"
    source: str = ""
    revision: str = ""
    artifact: str = ""
    target_model: str = ""
    metadata_model: str | None = None
    layer: int | None = None
    message: str | None = None


class TraceFrame(BaseModel):
    step: int = Field(ge=0)
    phase: Literal["prompt", "decode"]
    layer: int = Field(ge=0)
    token: str = ""
    generated_text: str = ""
    top_concepts: list[ConceptScore] = Field(default_factory=list)
    proxy_concepts: list[ConceptScore] = Field(default_factory=list)
    jacobian_concepts: list[ConceptScore] = Field(default_factory=list)
    jacobian_tokens: list[str] = Field(default_factory=list)
    layer_activations: list[LayerActivation] = Field(default_factory=list)
    action: FrameAction = "observe"
    note: str | None = None


class Intervention(BaseModel):
    step: int = Field(ge=0)
    from_concept: str
    to_concept: str
    source_token: str
    target_token: str
    strength: float
    source_rank_before: int = Field(ge=1)
    source_rank_after: int = Field(ge=1)
    target_rank_before: int = Field(ge=1)
    target_rank_after: int = Field(ge=1)
    source_probability_before: float = Field(ge=0.0, le=1.0)
    source_probability_after: float = Field(ge=0.0, le=1.0)
    target_probability_before: float = Field(ge=0.0, le=1.0)
    target_probability_after: float = Field(ge=0.0, le=1.0)
    jlens_kl_divergence: float = Field(ge=0.0)
    jlens_total_variation: float = Field(ge=0.0, le=1.0)
    reason: str


class JSpaceInjection(BaseModel):
    id: str
    keyword: str
    title: str
    source: str
    summary: str
    injected_text: str
    concept_targets: list[str] = Field(default_factory=list)
    target: InjectionTarget


class EngineInfo(BaseModel):
    mode: str
    model: str
    model_revision: str = ""
    layer: int
    device: str = "modal"
    readout: str
    jacobian_lens: JacobianLensInfo | None = None


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
    unsafe_words: list[str] = Field(default_factory=list, max_length=32)
    replacement_words: list[str] = Field(default_factory=list, max_length=32)
    rewrite_profile_id: str | None = None
    action: MonitorAction = "kill"
    threshold: float = Field(default=0.66, ge=0.05, le=0.99)
    lens_rank_threshold: int = Field(default=64, ge=1, le=512)
    lens_id: str = Field(default="anthropic-qwen3-1.7b-wikitext", min_length=1, max_length=120)
    max_new_tokens: int = Field(default=90, ge=8, le=256)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.05, le=1.0)
    strength: float = Field(default=1.0, ge=0.0, le=2.0)
    compare_baseline: bool = True
    seed: int = Field(default=7, ge=0)


class PolicyTermValidationRequest(BaseModel):
    terms: list[str] = Field(min_length=1, max_length=64)


class PolicyTermValidation(BaseModel):
    term: str
    valid: bool
    token_id: int | None = None
    decoded_token: str | None = None
    token_count: int = Field(ge=0)
    reason: str


class PolicyTermValidationResponse(BaseModel):
    tokenizer_model: str
    terms: list[PolicyTermValidation]


class RunResponse(BaseModel):
    id: str
    status: RunStatus
    mode: RunMode
    prompt: str
    output: str
    baseline_output: str | None = None
    decoder_output: str | None = None
    summary: str
    engine: EngineInfo
    frames: list[TraceFrame] = Field(default_factory=list)
    baseline_frames: list[TraceFrame] = Field(default_factory=list)
    decoder_frames: list[TraceFrame] = Field(default_factory=list)
    interventions: list[Intervention] = Field(default_factory=list)
    jspace_injections: list[JSpaceInjection] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)


class CustomConceptInput(BaseModel):
    label: str = Field(min_length=2, max_length=80)
    keywords: list[str] = Field(min_length=1, max_length=12)
    polarity: ConceptPolarity = "neutral"


class CustomConceptLens(BaseModel):
    id: str
    label: str
    polarity: ConceptPolarity
    keywords: list[str]
    calibration_prompts: list[str]
    proxy_score: float = Field(ge=0.0, le=1.0)
    jacobian_score: float = Field(ge=0.0, le=1.0)
    agreement_delta: float = Field(ge=0.0, le=1.0)


class CustomLensRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    concepts: list[CustomConceptInput] = Field(min_length=1, max_length=6)
    window_tokens: int = Field(default=64, ge=8, le=128)


class CustomLensResponse(BaseModel):
    id: str
    prompt: str
    summary: str
    engine: EngineInfo
    frame: TraceFrame
    concepts: list[CustomConceptLens] = Field(default_factory=list)
    jspace_injections: list[JSpaceInjection] = Field(default_factory=list)
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
    unsafe_words: list[str]
    replacement_words: list[str]
    default_strength: float
    demo_prompt: str
    control_phrases: list[str]


class ScenarioPublic(BaseModel):
    id: str
    mode: RunMode
    label: str
    prompt: str
    monitor_ids: list[str] = Field(default_factory=list)
    unsafe_words: list[str] = Field(default_factory=list)
    replacement_words: list[str] = Field(default_factory=list)
    rewrite_profile_id: str | None = None
    action: MonitorAction = "kill"
    threshold: float = 0.66
    lens_rank_threshold: int = Field(default=64, ge=1, le=512)
    max_new_tokens: int | None = Field(default=None, ge=1, le=256)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class LensPublic(BaseModel):
    id: str
    label: str
    description: str
    source: str
    revision: str
    artifact: str
    target_model: str
    dataset: str
    fitted_layers: int = Field(ge=1)


class ProfilesResponse(BaseModel):
    concepts: list[ConceptPublic]
    rewrite_profiles: list[RewriteProfilePublic]
    scenarios: list[ScenarioPublic]
    lenses: list[LensPublic]
    default_lens_id: str
