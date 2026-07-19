from __future__ import annotations

import os
import random
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .concepts import CONCEPTS, REWRITE_PROFILES, ConceptSpec, RewriteProfile
from .lenses import DEFAULT_LENS_ID, LENSES
from .lookups import lookup_jspace_injections
from .schemas import (
    ConceptPolarity,
    ConceptScore,
    CustomConceptInput,
    CustomConceptLens,
    CustomLensRequest,
    CustomLensResponse,
    EngineInfo,
    Intervention,
    JacobianLensInfo,
    JSpaceInjection,
    LayerActivation,
    PolicyTermValidation,
    PolicyTermValidationRequest,
    PolicyTermValidationResponse,
    RunMetrics,
    RunRequest,
    RunResponse,
    TraceFrame,
)


@dataclass
class _VectorBank:
    directions: dict[str, Any]
    target_layer: int
    jacobian_directions: dict[str, Any] | None = None


@dataclass
class _JacobianLensRuntime:
    id: str
    label: str
    lens_model: Any
    matrices: dict[int, Any]
    matrix_stack: Any
    source_layers: tuple[int, ...]
    source: str
    artifact: str
    target_model: str
    metadata_model: str | None
    layer: int
    d_model: int
    n_prompts: int


@dataclass
class _CustomConceptVector:
    id: str
    label: str
    polarity: ConceptPolarity
    keywords: list[str]
    calibration_prompts: list[str]
    proxy_direction: Any
    jacobian_direction: Any | None


class HFJSpaceEngine:
    """Qwen hidden-state monitor and activation rewrite engine.

    Monitoring uses Anthropic's published Jacobian lens directly: each fitted layer is
    transported into the final residual basis and decoded with Qwen's own unembedding.
    Rewrite uses those vocabulary ranks as triggers and swaps the matched unsafe token's
    J-lens coordinate with an approved token coordinate at the selected source layer.
    """

    mode = "hf"
    readout = "Anthropic J-lens rank filter + coordinate swaps"

    def __init__(self) -> None:
        self.lens_id = os.getenv("JSTEER_LENS_ID", DEFAULT_LENS_ID)
        lens_spec = LENSES.get(self.lens_id)
        if lens_spec is None:
            available = ", ".join(sorted(LENSES))
            raise ValueError(f"Unsupported JSTEER_LENS_ID={self.lens_id!r}. Available lenses: {available}.")
        self.model_name = os.getenv("JSTEER_MODEL", "Qwen/Qwen3-1.7B")
        self.lens_label = lens_spec.label
        self.lens_source = os.getenv("JSTEER_JLENS_REPO", lens_spec.source)
        self.lens_artifact = os.getenv(
            "JSTEER_JLENS_FILE",
            lens_spec.artifact,
        )
        self.lens_target_model = os.getenv("JSTEER_JLENS_MODEL", lens_spec.target_model)
        self._loaded = False
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.torch: Any | None = None
        self.device = "unloaded"
        self.layer = int(os.getenv("JSTEER_LAYER", "-1"))
        self.bank: _VectorBank | None = None
        self.jacobian_lens: _JacobianLensRuntime | None = None
        self.jacobian_lens_error: str | None = None
        self._concept_token_cache: dict[str, list[int]] = {}
        self._lens_filter_token_cache: dict[str, dict[int, str]] = {}

    def info(self) -> EngineInfo:
        return EngineInfo(
            mode=self.mode,
            model=self.model_name,
            layer=max(self.layer, 0),
            device=self.device,
            readout=self.readout,
            jacobian_lens=self._jacobian_lens_info(),
        )

    def run(self, request: RunRequest) -> RunResponse:
        self._validate_lens_selection(request.lens_id)
        self._ensure_loaded()
        self._require_valid_policy_terms(request)
        started = time.perf_counter()
        if request.mode == "rewrite":
            response = self._run_rewrite(request)
        else:
            response = self._run_monitor(request)
        response.metrics.latency_ms = int((time.perf_counter() - started) * 1000)
        return response

    def stream_run(self, request: RunRequest) -> Iterator[dict[str, Any]]:
        self._validate_lens_selection(request.lens_id)
        self._ensure_loaded()
        self._require_valid_policy_terms(request)
        started = time.perf_counter()
        run_id = str(uuid.uuid4())
        yield {"type": "start", "response": self._running_response(run_id, request)}
        if request.mode == "rewrite":
            yield from self._stream_rewrite(request, run_id, started)
        else:
            yield from self._stream_monitor(request, run_id, started)

    def validate_policy_terms(
        self,
        request: PolicyTermValidationRequest,
    ) -> PolicyTermValidationResponse:
        self._ensure_loaded()
        return PolicyTermValidationResponse(
            tokenizer_model=self.model_name,
            terms=[self._validate_policy_term(term) for term in request.terms],
        )

    def _require_valid_policy_terms(self, request: RunRequest) -> None:
        terms = [*request.unsafe_words, *request.replacement_words]
        invalid = [item for item in (self._validate_policy_term(term) for term in terms) if not item.valid]
        if invalid:
            details = "; ".join(f"{item.term!r}: {item.reason}" for item in invalid)
            raise ValueError(f"Policy terms must each map to one Qwen token. {details}")

    def _validate_policy_term(self, raw_term: str) -> PolicyTermValidation:
        assert self.tokenizer is not None
        term = raw_term.strip()
        if not term:
            return PolicyTermValidation(
                term=raw_term,
                valid=False,
                token_count=0,
                reason="Empty terms are not valid policy words.",
            )
        single_token_ids = self._single_token_ids_for_term(term)
        if single_token_ids:
            token_id = single_token_ids[0]
            return PolicyTermValidation(
                term=term,
                valid=True,
                token_id=token_id,
                decoded_token=self._decode_token(token_id),
                token_count=1,
                reason="Exact single-token Qwen vocabulary entry.",
            )
        token_count = len(self.tokenizer.encode(term, add_special_tokens=False))
        return PolicyTermValidation(
            term=term,
            valid=False,
            token_count=token_count,
            reason=f"Qwen splits this term into {token_count} tokens.",
        )

    def _running_response(self, run_id: str, request: RunRequest) -> RunResponse:
        return RunResponse(
            id=run_id,
            status="running",
            mode=request.mode,
            prompt=request.prompt,
            output="",
            baseline_output=None,
            summary="Run streaming token readouts from the active decoder.",
            engine=self.info(),
            frames=[],
            baseline_frames=[],
            interventions=[],
            metrics=RunMetrics(),
        )

    def custom_lens(self, request: CustomLensRequest) -> CustomLensResponse:
        self._ensure_loaded()
        assert self.model is not None and self.tokenizer is not None
        started = time.perf_counter()
        injections = lookup_jspace_injections(
            request.prompt,
            target="custom_lens",
            custom_concepts=request.concepts,
        )
        custom_bank = self._build_custom_lens_bank(request.concepts, injections)
        input_ids = self._encode_chat(request.prompt)
        outputs = self._forward(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[self.layer + 1][0]
        tail_count = min(request.window_tokens, int(hidden_states.shape[0]))
        tail = hidden_states[-tail_count:]

        proxy_scores = self._score_custom_proxy_sequence(tail, custom_bank)
        jacobian_scores, jacobian_tokens = self._score_custom_jacobian_sequence(tail, custom_bank)
        top_proxy = proxy_scores[0] if proxy_scores else None
        top_jacobian = jacobian_scores[0] if jacobian_scores else None
        alignment = "aligned" if top_proxy and top_jacobian and top_proxy.id == top_jacobian.id else "diverged"
        frame = TraceFrame(
            step=0,
            phase="prompt",
            layer=self.layer,
            token=f"<last {tail_count} tokens>",
            top_concepts=proxy_scores,
            proxy_concepts=proxy_scores,
            jacobian_concepts=jacobian_scores,
            jacobian_tokens=jacobian_tokens,
            layer_activations=self._custom_layer_activation_scan(
                outputs.hidden_states,
                custom_bank,
                prompt_window=min(4, tail_count),
            ),
            action="observe",
            note=self._custom_lens_note(injections),
        )
        score_by_id = {score.id: score.score for score in proxy_scores}
        jacobian_by_id = {score.id: score.score for score in jacobian_scores}
        concepts = [
            CustomConceptLens(
                id=item.id,
                label=item.label,
                polarity=item.polarity,
                keywords=item.keywords,
                calibration_prompts=item.calibration_prompts,
                proxy_score=round(score_by_id.get(item.id, 0.0), 3),
                jacobian_score=round(jacobian_by_id.get(item.id, 0.0), 3),
                agreement_delta=round(abs(score_by_id.get(item.id, 0.0) - jacobian_by_id.get(item.id, 0.0)), 3),
            )
            for item in custom_bank
        ]
        return CustomLensResponse(
            id=str(uuid.uuid4()),
            prompt=request.prompt,
            summary=self._custom_lens_summary(top_proxy, top_jacobian, alignment, len(custom_bank), len(injections)),
            engine=self.info(),
            frame=frame,
            concepts=concepts,
            jspace_injections=injections,
            metrics=RunMetrics(
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt_tokens=int(input_ids.shape[-1]),
                peak_score=max((score.score for score in proxy_scores), default=0.0),
            ),
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "HF engine dependencies are missing. Run `uv sync --group dev` before serving."
            ) from exc

        self.torch = torch
        dtype = "auto"
        device_map: str | None = "auto"
        if os.getenv("JSTEER_DEVICE", "").lower() == "cpu":
            device_map = None

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        if device_map is None:
            self.model.to("cpu")
        self.model.eval()
        self.device = str(next(self.model.parameters()).device)

        layer_count = len(self.model.model.layers)
        if self.layer < 0:
            self.layer = max(1, int(layer_count * 0.58))
        self._load_jacobian_lens()
        self._loaded = True

    def _jacobian_lens_info(self) -> JacobianLensInfo:
        if self.jacobian_lens is not None:
            return JacobianLensInfo(
                id=self.jacobian_lens.id,
                label=self.jacobian_lens.label,
                available=True,
                status="ready",
                source=self.jacobian_lens.source,
                artifact=self.jacobian_lens.artifact,
                target_model=self.jacobian_lens.target_model,
                metadata_model=self.jacobian_lens.metadata_model,
                layer=self.jacobian_lens.layer,
                message=(
                    f"Loaded d_model={self.jacobian_lens.d_model} lens from "
                    f"{self.jacobian_lens.n_prompts} prompts."
                ),
            )
        if self.jacobian_lens_error:
            return JacobianLensInfo(
                id=self.lens_id,
                label=self.lens_label,
                available=False,
                status="unavailable",
                source=self.lens_source,
                artifact=self.lens_artifact,
                target_model=self.lens_target_model,
                metadata_model=None,
                layer=max(self.layer, 0),
                message=self.jacobian_lens_error,
            )
        return JacobianLensInfo(
            id=self.lens_id,
            label=self.lens_label,
            available=False,
            status="configured",
            source=self.lens_source,
            artifact=self.lens_artifact,
            target_model=self.lens_target_model,
            metadata_model=None,
            layer=max(self.layer, 0),
            message="Lens will load with the first model-backed run.",
        )

    def _load_jacobian_lens(self) -> None:
        assert self.model is not None and self.tokenizer is not None and self.torch is not None
        try:
            import jlens
        except ImportError as exc:
            self.jacobian_lens_error = f"jlens package is not installed: {exc}"
            return

        try:
            if not self._model_ids_match(self.model_name, self.lens_target_model):
                self.jacobian_lens_error = (
                    f"Configured lens target {self.lens_target_model} does not match loaded model "
                    f"{self.model_name}. Set JSTEER_MODEL and JSTEER_JLENS_MODEL to the same checkpoint."
                )
                return
            metadata_model = self._lens_metadata_model()
            if metadata_model and not self._model_ids_match(metadata_model, self.lens_target_model):
                self.jacobian_lens_error = (
                    f"Lens metadata names {metadata_model}, but JSTEER_JLENS_MODEL is "
                    f"{self.lens_target_model}. Use the lens artifact fitted for the loaded Qwen checkpoint."
                )
                return
            lens = jlens.JacobianLens.from_pretrained(self.lens_source, filename=self.lens_artifact)
            text_config = self.model.config.get_text_config()
            hidden_size = int(text_config.hidden_size)
            if int(lens.d_model) != hidden_size:
                self.jacobian_lens_error = (
                    f"Lens d_model={lens.d_model} does not match {self.model_name} hidden_size={hidden_size}. "
                    "Use a lens artifact fitted for the exact Qwen checkpoint."
                )
                return
            if self.layer not in lens.source_layers:
                self.jacobian_lens_error = (
                    f"Lens has layers {lens.source_layers[0]}..{lens.source_layers[-1]}, "
                    f"but JSTEER_LAYER={self.layer} is not present."
                )
                return

            lens_model = jlens.from_hf(self.model, self.tokenizer, force_bos=False)
            parameter = next(self.model.parameters())
            source_layers = tuple(int(layer) for layer in lens.source_layers)
            matrix_stack = self.torch.stack(
                [
                    lens.jacobians[layer].to(device=parameter.device, dtype=parameter.dtype)
                    for layer in source_layers
                ]
            )
            matrices = {layer: matrix_stack[index] for index, layer in enumerate(source_layers)}
            self.jacobian_lens = _JacobianLensRuntime(
                id=self.lens_id,
                label=self.lens_label,
                lens_model=lens_model,
                matrices=matrices,
                matrix_stack=matrix_stack,
                source_layers=source_layers,
                source=self.lens_source,
                artifact=self.lens_artifact,
                target_model=self.lens_target_model,
                metadata_model=metadata_model,
                layer=self.layer,
                d_model=int(lens.d_model),
                n_prompts=int(lens.n_prompts),
            )
            self.jacobian_lens_error = None
        except Exception as exc:
            self.jacobian_lens_error = f"Unable to load Jacobian lens: {exc}"

    def _validate_lens_selection(self, lens_id: str) -> None:
        if lens_id != self.lens_id:
            raise ValueError(
                f"Lens {lens_id!r} is not loaded by this runtime. Select the compatible lens {self.lens_id!r}."
            )

    @classmethod
    def _model_ids_match(cls, model_name: str, lens_target_model: str) -> bool:
        return cls._normalize_model_id(model_name) == cls._normalize_model_id(lens_target_model)

    @staticmethod
    def _normalize_model_id(model_name: str) -> str:
        return model_name.strip().strip("/").lower()

    def _lens_metadata_model(self) -> str | None:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(self.lens_source, self._lens_config_filename(self.lens_artifact))
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if stripped.startswith("hf_model_name:"):
                        return stripped.split(":", 1)[1].strip().strip('"').strip("'")
        except Exception:
            return None
        return None

    @staticmethod
    def _lens_config_filename(lens_artifact: str) -> str:
        directory = lens_artifact.rsplit("/", 1)[0] if "/" in lens_artifact else ""
        return f"{directory}/config.yaml" if directory else "config.yaml"

    def _monitor_prompt_frame(
        self,
        hidden_states: Any,
        monitor_ids: list[str],
        rank_threshold: int,
        unsafe_words: list[str] | None = None,
    ) -> TraceFrame:
        layer_readouts = self._jlens_layer_readouts(
            hidden_states,
            monitor_ids,
            rank_threshold,
            unsafe_words,
        )
        scores = self._lens_match_scores(layer_readouts)
        control_readout = min(
            layer_readouts,
            key=lambda item: abs(item.layer - self.layer),
            default=None,
        )
        return TraceFrame(
            step=0,
            phase="prompt",
            layer=self.layer,
            token="<prompt>",
            top_concepts=scores,
            proxy_concepts=[],
            jacobian_concepts=scores,
            jacobian_tokens=control_readout.top_tokens if control_readout else [],
            layer_activations=layer_readouts,
            action="observe",
            note="Prompt boundary decoded directly through Anthropic's per-layer Jacobian lens.",
        )

    def _run_monitor(self, request: RunRequest) -> RunResponse:
        assert self.model is not None and self.tokenizer is not None
        monitor_ids = request.monitor_ids or [
            "credential_exfiltration",
            "harassment",
            "sexual_content",
            "unsupported_claim",
        ]
        input_ids = self._encode_chat(request.prompt)
        outputs = self._forward(input_ids, output_hidden_states=True)
        prompt_frame = self._monitor_prompt_frame(
            outputs.hidden_states,
            monitor_ids,
            request.lens_rank_threshold,
            request.unsafe_words or None,
        )
        peak_score = self._frame_peak(prompt_frame)
        frames = [prompt_frame]
        prompt_match = self._best_lens_match(prompt_frame)
        if request.action == "kill" and prompt_match is not None:
            prompt_frame.action = "blocked"
            prompt_frame.note = "Unsafe vocabulary entered the Anthropic J-lens rank cutoff before decoding."
            return RunResponse(
                id=str(uuid.uuid4()),
                status="blocked",
                mode="monitor",
                prompt=request.prompt,
                output="",
                summary=(
                    f"Run killed before output: {prompt_match.label} matched J-lens word "
                    f"{prompt_match.matched_token!r} at rank {prompt_match.rank} on layer {prompt_match.layer}."
                ),
                engine=self.info(),
                frames=frames,
                metrics=RunMetrics(
                    prompt_tokens=int(input_ids.shape[-1]),
                    completion_tokens=0,
                    peak_score=peak_score,
                ),
            )

        text, decode_frames = self._generate(
            input_ids=input_ids,
            request=request,
            monitor_ids=monitor_ids,
            steering=None,
        )
        if request.action == "kill":
            for index, frame in enumerate(decode_frames):
                match = self._best_lens_match(frame)
                if frame.action == "blocked" and match is not None:
                    blocked_frames = frames + decode_frames[: index + 1]
                    partial_text = frame.generated_text
                    return RunResponse(
                        id=str(uuid.uuid4()),
                        status="blocked",
                        mode="monitor",
                        prompt=request.prompt,
                        output=partial_text,
                        summary=(
                            f"Run killed during decode: {match.label} matched J-lens word {match.matched_token!r} "
                            f"at rank {match.rank} on layer {match.layer}, before step {frame.step}."
                        ),
                        engine=self.info(),
                        frames=blocked_frames,
                        metrics=RunMetrics(
                            prompt_tokens=int(input_ids.shape[-1]),
                            completion_tokens=len(self.tokenizer.encode(partial_text, add_special_tokens=False)),
                            peak_score=max(self._frame_peak(item) for item in blocked_frames),
                        ),
                    )
        frames.extend(decode_frames)
        return RunResponse(
            id=str(uuid.uuid4()),
            status="completed",
            mode="monitor",
            prompt=request.prompt,
            output=text,
            summary=self._monitor_completion_summary(frames),
            engine=self.info(),
            frames=frames,
            metrics=RunMetrics(
                prompt_tokens=int(input_ids.shape[-1]),
                completion_tokens=len(self.tokenizer.encode(text, add_special_tokens=False)),
                peak_score=max([peak_score, *[self._frame_peak(frame) for frame in decode_frames]]),
            ),
        )

    def _stream_monitor(
        self,
        request: RunRequest,
        run_id: str,
        started: float,
    ) -> Iterator[dict[str, Any]]:
        assert self.model is not None and self.tokenizer is not None
        monitor_ids = request.monitor_ids or [
            "credential_exfiltration",
            "harassment",
            "sexual_content",
            "unsupported_claim",
        ]
        input_ids = self._encode_chat(request.prompt)
        outputs = self._forward(input_ids, output_hidden_states=True)
        prompt_frame = self._monitor_prompt_frame(
            outputs.hidden_states,
            monitor_ids,
            request.lens_rank_threshold,
            request.unsafe_words or None,
        )
        peak_score = self._frame_peak(prompt_frame)
        frames = [prompt_frame]

        prompt_match = self._best_lens_match(prompt_frame)
        if request.action == "kill" and prompt_match is not None:
            prompt_frame.action = "blocked"
            prompt_frame.note = "Unsafe vocabulary entered the Anthropic J-lens rank cutoff before decoding."
            yield {"type": "frame", "frame": prompt_frame}
            yield {
                "type": "done",
                "response": RunResponse(
                    id=run_id,
                    status="blocked",
                    mode="monitor",
                    prompt=request.prompt,
                    output="",
                    summary=(
                        f"Run killed before output: {prompt_match.label} matched J-lens word "
                        f"{prompt_match.matched_token!r} at rank {prompt_match.rank} on layer {prompt_match.layer}."
                    ),
                    engine=self.info(),
                    frames=frames,
                    metrics=RunMetrics(
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        prompt_tokens=int(input_ids.shape[-1]),
                        completion_tokens=0,
                        peak_score=peak_score,
                    ),
                ),
            }
            return

        yield {"type": "frame", "frame": prompt_frame}

        text = ""
        for frame in self._iter_generate(
            input_ids=input_ids,
            request=request,
            monitor_ids=monitor_ids,
            steering=None,
        ):
            text = frame.generated_text
            match = self._best_lens_match(frame)
            if request.action == "kill" and frame.action == "blocked" and match is not None:
                frames.append(frame)
                yield {"type": "frame", "frame": frame}
                yield {
                    "type": "done",
                    "response": RunResponse(
                        id=run_id,
                        status="blocked",
                        mode="monitor",
                        prompt=request.prompt,
                        output=text,
                        summary=(
                            f"Run killed during decode: {match.label} matched J-lens word {match.matched_token!r} "
                            f"at rank {match.rank} on layer {match.layer}, before step {frame.step}."
                        ),
                        engine=self.info(),
                        frames=frames,
                        metrics=RunMetrics(
                            latency_ms=int((time.perf_counter() - started) * 1000),
                            prompt_tokens=int(input_ids.shape[-1]),
                            completion_tokens=len(self.tokenizer.encode(text, add_special_tokens=False)),
                            peak_score=max(self._frame_peak(item) for item in frames),
                        ),
                    ),
                }
                return
            frames.append(frame)
            yield {"type": "frame", "frame": frame}

        yield {
            "type": "done",
            "response": RunResponse(
                id=run_id,
                status="completed",
                mode="monitor",
                prompt=request.prompt,
                output=text,
                summary=self._monitor_completion_summary(frames),
                engine=self.info(),
                frames=frames,
                metrics=RunMetrics(
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    prompt_tokens=int(input_ids.shape[-1]),
                    completion_tokens=len(self.tokenizer.encode(text, add_special_tokens=False)),
                    peak_score=max(self._frame_peak(frame) for frame in frames),
                ),
            ),
        }

    def _run_rewrite(self, request: RunRequest) -> RunResponse:
        assert self.model is not None and self.tokenizer is not None
        profile = REWRITE_PROFILES.get(request.rewrite_profile_id or "", REWRITE_PROFILES["brand_canon_injection"])
        injections = lookup_jspace_injections(
            request.prompt,
            target="rewrite_controller",
            default_targets=[CONCEPTS[profile.target_concept].label],
        )
        input_ids = self._encode_chat(request.prompt)
        monitor_ids = list(profile.suppress_concepts) + [profile.target_concept]

        baseline_text = None
        baseline_frames: list[TraceFrame] = []
        decoder_text = None
        decoder_frames: list[TraceFrame] = []
        if request.compare_baseline:
            baseline_text, baseline_frames = self._generate(
                input_ids=input_ids,
                request=request,
                monitor_ids=monitor_ids,
                steering=None,
            )
            decoder_state = self._decoder_control_state(profile, request, injections)
            decoder_text, decoder_frames = self._generate(
                input_ids=input_ids,
                request=request,
                monitor_ids=monitor_ids,
                steering=decoder_state,
            )

        initial_match = self._rewrite_prompt_match(
            input_ids,
            list(profile.suppress_concepts),
            request.lens_rank_threshold,
            request.unsafe_words or None,
        )
        steering = self._steering_state(profile, request, injections, initial_match)
        steered_text, frames = self._generate(
            input_ids=input_ids,
            request=request,
            monitor_ids=monitor_ids,
            steering=steering,
        )
        interventions = steering["interventions"]
        peak = max((self._frame_peak(frame) for frame in frames), default=0.0)

        return RunResponse(
            id=str(uuid.uuid4()),
            status="rewritten",
            mode="rewrite",
            prompt=request.prompt,
            output=steered_text,
            baseline_output=baseline_text,
            decoder_output=decoder_text,
            summary=f"{len(interventions)} J-lens-triggered rewrites applied at layer {self.layer}.",
            engine=self.info(),
            frames=frames,
            baseline_frames=baseline_frames,
            decoder_frames=decoder_frames,
            interventions=interventions,
            jspace_injections=injections,
            metrics=RunMetrics(
                prompt_tokens=int(input_ids.shape[-1]),
                completion_tokens=len(self.tokenizer.encode(steered_text, add_special_tokens=False)),
                peak_score=peak,
                rewrites=len(interventions),
            ),
        )

    def _stream_rewrite(
        self,
        request: RunRequest,
        run_id: str,
        started: float,
    ) -> Iterator[dict[str, Any]]:
        assert self.model is not None and self.tokenizer is not None
        profile = REWRITE_PROFILES.get(request.rewrite_profile_id or "", REWRITE_PROFILES["brand_canon_injection"])
        injections = lookup_jspace_injections(
            request.prompt,
            target="rewrite_controller",
            default_targets=[CONCEPTS[profile.target_concept].label],
        )
        input_ids = self._encode_chat(request.prompt)
        monitor_ids = list(profile.suppress_concepts) + [profile.target_concept]

        baseline_text = None
        baseline_frames: list[TraceFrame] = []
        decoder_text = None
        decoder_frames: list[TraceFrame] = []
        if request.compare_baseline:
            for frame in self._iter_generate(
                input_ids=input_ids,
                request=request,
                monitor_ids=monitor_ids,
                steering=None,
            ):
                baseline_frames.append(frame)
                baseline_text = frame.generated_text
                yield {"type": "baseline_frame", "frame": frame}
            decoder_state = self._decoder_control_state(profile, request, injections)
            for frame in self._iter_generate(
                input_ids=input_ids,
                request=request,
                monitor_ids=monitor_ids,
                steering=decoder_state,
            ):
                decoder_frames.append(frame)
                decoder_text = frame.generated_text
                yield {"type": "decoder_frame", "frame": frame}

        initial_match = self._rewrite_prompt_match(
            input_ids,
            list(profile.suppress_concepts),
            request.lens_rank_threshold,
            request.unsafe_words or None,
        )
        steering = self._steering_state(profile, request, injections, initial_match)
        frames: list[TraceFrame] = []
        steered_text = ""
        for frame in self._iter_generate(
            input_ids=input_ids,
            request=request,
            monitor_ids=monitor_ids,
            steering=steering,
        ):
            frames.append(frame)
            steered_text = frame.generated_text
            yield {"type": "frame", "frame": frame}

        interventions = steering["interventions"]
        peak = max((self._frame_peak(frame) for frame in frames), default=0.0)
        yield {
            "type": "done",
            "response": RunResponse(
                id=run_id,
                status="rewritten",
                mode="rewrite",
                prompt=request.prompt,
                output=steered_text,
                baseline_output=baseline_text,
                decoder_output=decoder_text,
                summary=f"{len(interventions)} J-lens-triggered rewrites applied at layer {self.layer}.",
                engine=self.info(),
                frames=frames,
                baseline_frames=baseline_frames,
                decoder_frames=decoder_frames,
                interventions=interventions,
                jspace_injections=injections,
                metrics=RunMetrics(
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    prompt_tokens=int(input_ids.shape[-1]),
                    completion_tokens=len(self.tokenizer.encode(steered_text, add_special_tokens=False)),
                    peak_score=peak,
                    rewrites=len(interventions),
                ),
            ),
        }

    def _generate(
        self,
        input_ids: Any,
        request: RunRequest,
        monitor_ids: list[str],
        steering: dict[str, Any] | None,
    ) -> tuple[str, list[TraceFrame]]:
        frames = list(
            self._iter_generate(
                input_ids=input_ids,
                request=request,
                monitor_ids=monitor_ids,
                steering=steering,
            )
        )
        text = frames[-1].generated_text.strip() if frames else ""
        return text, frames

    def _iter_generate(
        self,
        input_ids: Any,
        request: RunRequest,
        monitor_ids: list[str],
        steering: dict[str, Any] | None,
    ) -> Iterator[TraceFrame]:
        assert self.model is not None and self.tokenizer is not None and self.torch is not None
        torch = self.torch
        handle = None
        state = steering
        if state is not None and state.get("activation_edit"):
            handle = self.model.model.layers[self.layer].register_forward_hook(
                lambda module, inputs, output: self._steering_hook(output, state)
            )

        try:
            generated_ids: list[int] = []
            past_key_values = None
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
            current_ids = input_ids
            for step in range(request.max_new_tokens):
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=current_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                past_key_values = outputs.past_key_values
                layer_readouts = self._jlens_layer_readouts(
                    outputs.hidden_states,
                    monitor_ids,
                    request.lens_rank_threshold,
                    request.unsafe_words or None,
                )
                scores = self._lens_match_scores(layer_readouts)
                proxy_scores: list[ConceptScore] = []
                jacobian_scores = scores
                control_readout = min(
                    layer_readouts,
                    key=lambda item: abs(item.layer - self.layer),
                    default=None,
                )
                jacobian_tokens = control_readout.top_tokens if control_readout else []
                next_match = self._best_ranked_readout(layer_readouts)
                if request.mode == "monitor":
                    if request.action == "kill" and step > 0 and next_match is not None:
                        yield TraceFrame(
                            step=step + 1,
                            phase="decode",
                            layer=self.layer,
                            token="",
                            generated_text=self.tokenizer.decode(generated_ids, skip_special_tokens=True),
                            top_concepts=scores,
                            proxy_concepts=[],
                            jacobian_concepts=scores,
                            jacobian_tokens=jacobian_tokens,
                            layer_activations=layer_readouts,
                            action="blocked",
                            note="Unsafe vocabulary entered the Anthropic J-lens rank cutoff before the next token.",
                        )
                        return
                else:
                    if state is not None and state.get("activation_edit"):
                        state["pending_match"] = next_match
                logits = outputs.logits[:, -1, :]
                if state is not None and state.get("decoder_constraint") and state.get("has_rewritten"):
                    logits = self._apply_control_phrase_bias(logits, state, generated_ids)
                next_token = self._sample(logits, request.temperature, request.top_p, request.seed + step)
                token_id = int(next_token.item())
                if token_id == self.tokenizer.eos_token_id:
                    break
                generated_ids.append(token_id)
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                action = "rewrite" if state and state.pop("last_rewrite", False) else "observe"
                rewrite_match = state.pop("last_match", None) if state and action == "rewrite" else None
                yield TraceFrame(
                    step=step + 1,
                    phase="decode",
                    layer=self.layer,
                    token=self.tokenizer.decode([token_id], skip_special_tokens=True),
                    generated_text=generated_text,
                    top_concepts=scores,
                    proxy_concepts=proxy_scores,
                    jacobian_concepts=jacobian_scores,
                    jacobian_tokens=jacobian_tokens,
                    layer_activations=layer_readouts,
                    action=action,
                    note=(
                        (
                            f"J-lens matched {rewrite_match.matched_token!r} at "
                            f"L{rewrite_match.layer} rank {rewrite_match.rank}; its coordinate "
                            "was swapped with an approved token at the control layer."
                        )
                        if rewrite_match is not None
                        else "J-lens coordinate swap edited the control layer."
                        if action == "rewrite"
                        else None
                    ),
                )
                if (
                    state is not None
                    and state.get("decoder_constraint")
                    and self._control_phrase_complete(state, generated_ids, generated_text)
                ):
                    break
                current_ids = next_token.reshape(1, 1)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ],
                    dim=-1,
                )
        finally:
            if handle is not None:
                handle.remove()

    def _steering_state(
        self,
        profile: RewriteProfile,
        request: RunRequest,
        lookup_injections: list[JSpaceInjection] | None = None,
        initial_match: LayerActivation | None = None,
    ) -> dict[str, Any]:
        assert self.jacobian_lens is not None and self.tokenizer is not None and self.torch is not None
        phrase_sequences = self._control_phrase_sequences(profile)
        return {
            "profile": profile,
            "lookup_injections": lookup_injections or [],
            "phrase_sequences": phrase_sequences,
            "phrase_bias": max(0.0, 4.0 + request.strength * 2.2),
            "force_control_phrase": True,
            "activation_edit": True,
            "decoder_constraint": False,
            "target_token_ids": self._rewrite_target_token_ids(
                profile,
                lookup_injections or [],
                request.replacement_words,
            ),
            "suppress_token_ids": self._rewrite_suppress_token_ids(profile, request.unsafe_words),
            "strength": request.strength,
            "pending_match": initial_match,
            "interventions": [],
            "last_rewrite": False,
            "last_match": None,
            "has_rewritten": False,
            "step": 0,
        }

    def _decoder_control_state(
        self,
        profile: RewriteProfile,
        request: RunRequest,
        lookup_injections: list[JSpaceInjection] | None = None,
    ) -> dict[str, Any]:
        state = self._steering_state(profile, request, lookup_injections, initial_match=None)
        state.update(
            {
                "activation_edit": False,
                "decoder_constraint": True,
                "has_rewritten": True,
            }
        )
        return state

    def _steering_hook(self, output: Any, state: dict[str, Any]) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        last = hidden[0, -1, :].detach()
        state["step"] += 1
        match: LayerActivation | None = state.pop("pending_match", None)
        state["last_rewrite"] = match is not None
        state["last_match"] = match
        if match is None:
            return output

        source_token_id = self._matched_policy_token_id(match, state["suppress_token_ids"])
        delta, target_token_id = self._jlens_coordinate_swap_delta(
            last,
            source_token_id,
            state["target_token_ids"],
            state["strength"],
        )
        delta = delta.to(device=hidden.device, dtype=hidden.dtype)
        edited = hidden.clone()
        edited[:, -1, :] = edited[:, -1, :] + delta
        state["has_rewritten"] = True
        profile: RewriteProfile = state["profile"]
        lookup_injections: list[JSpaceInjection] = state.get("lookup_injections", [])
        source_token = self._decode_token(source_token_id)
        target_token = self._decode_token(target_token_id)
        state["interventions"].append(
            Intervention(
                step=max(0, state["step"] - 1),
                from_concept=match.label,
                to_concept=CONCEPTS[profile.target_concept].label,
                strength=float(delta.norm().detach().float().cpu()),
                reason=(
                    f"Anthropic J-lens word {match.matched_token!r} entered the rank cutoff at "
                    f"L{match.layer} rank {match.rank}. Its {source_token!r} coordinate was swapped "
                    f"with the approved {target_token!r} coordinate at the control layer, while the "
                    "residual component outside their two-direction span was preserved."
                    + (
                        " Retrieved approved terms were available to the target selector."
                        if lookup_injections
                        else ""
                    )
                ),
            )
        )
        if isinstance(output, tuple):
            return (edited, *output[1:])
        return edited

    def _rewrite_prompt_match(
        self,
        input_ids: Any,
        concept_ids: list[str],
        rank_threshold: int,
        unsafe_words: list[str] | None = None,
    ) -> LayerActivation | None:
        outputs = self._forward(input_ids, output_hidden_states=True)
        readouts = self._jlens_layer_readouts(
            outputs.hidden_states,
            concept_ids,
            rank_threshold,
            unsafe_words,
        )
        return self._best_ranked_readout(readouts)

    def _rewrite_target_token_ids(
        self,
        profile: RewriteProfile,
        lookup_injections: list[JSpaceInjection],
        replacement_words: list[str] | None = None,
    ) -> list[int]:
        assert self.tokenizer is not None
        token_ids = (
            self._policy_term_token_ids(replacement_words)
            if replacement_words
            else list(self._single_token_policy_terms(CONCEPTS[profile.target_concept]))
        )
        texts = [item.injected_text for item in lookup_injections[:3]]
        for text in texts:
            for term in text.replace(".", " ").replace(",", " ").split():
                for variant in (term, f" {term}"):
                    encoded = self.tokenizer.encode(variant, add_special_tokens=False)
                    if len(encoded) == 1:
                        token_ids.append(int(encoded[0]))
        return list(dict.fromkeys(token_ids))[:96]

    def _rewrite_suppress_token_ids(
        self,
        profile: RewriteProfile,
        unsafe_words: list[str] | None = None,
    ) -> list[int]:
        if unsafe_words:
            return self._policy_term_token_ids(unsafe_words)
        token_ids = [
            token_id
            for concept_id in profile.suppress_concepts
            for token_id in self._single_token_policy_terms(CONCEPTS[concept_id])
        ]
        return list(dict.fromkeys(token_ids))

    def _policy_term_token_ids(self, terms: list[str]) -> list[int]:
        token_ids = [
            token_id
            for term in terms
            for token_id in self._single_token_ids_for_term(term)
        ]
        return list(dict.fromkeys(token_ids))

    def _single_token_ids_for_term(self, raw_term: str) -> list[int]:
        assert self.tokenizer is not None
        term = raw_term.strip()
        if not term:
            return []
        token_ids: list[int] = []
        for variant in (term, f" {term}"):
            encoded = self.tokenizer.encode(variant, add_special_tokens=False)
            if len(encoded) == 1:
                token_ids.append(int(encoded[0]))
        return list(dict.fromkeys(token_ids))

    def _matched_policy_token_id(self, match: LayerActivation, suppress_token_ids: list[int]) -> int:
        if not suppress_token_ids:
            raise RuntimeError("The rewrite profile has no Qwen source tokens for J-lens steering.")
        matched = (match.matched_token or "").strip().casefold()
        for token_id in suppress_token_ids:
            if self._decode_token(token_id).casefold() == matched:
                return token_id
        return suppress_token_ids[0]

    def _decode_token(self, token_id: int) -> str:
        assert self.tokenizer is not None
        return self.tokenizer.decode([token_id], skip_special_tokens=True).strip() or f"token {token_id}"

    def _jlens_coordinate_swap_delta(
        self,
        hidden: Any,
        source_token_id: int,
        target_token_ids: list[int],
        scale: float,
    ) -> tuple[Any, int]:
        """Swap source/target J-lens coordinates while preserving the orthogonal residual."""
        assert self.jacobian_lens is not None and self.torch is not None
        if not target_token_ids:
            raise RuntimeError("The rewrite profile has no Qwen target tokens for J-lens steering.")

        matrix = self.jacobian_lens.matrices[self.layer]
        with self.torch.enable_grad():
            source = hidden.detach().to(device=matrix.device, dtype=matrix.dtype).requires_grad_(True)
            transported = source @ matrix.T
            logits = self.jacobian_lens.lens_model.unembed(transported.unsqueeze(0))[0]
            target_index = int(logits[target_token_ids].argmax().detach().cpu())
            target_token_id = target_token_ids[target_index]
            (source_direction,) = self.torch.autograd.grad(
                logits[source_token_id], source, retain_graph=True
            )
            (target_direction,) = self.torch.autograd.grad(logits[target_token_id], source)

        source_direction = source_direction.detach()
        target_direction = target_direction.detach()
        source_direction = source_direction / (source_direction.norm() + 1e-6)
        target_direction = target_direction / (target_direction.norm() + 1e-6)
        basis = self.torch.stack((source_direction, target_direction), dim=1)
        coordinates = self.torch.linalg.pinv(basis.float()) @ source.detach().float()
        swapped = coordinates.flip(0)
        delta = float(scale) * (basis.float() @ (swapped - coordinates))
        return delta.to(dtype=hidden.dtype), target_token_id

    def _control_phrase_sequences(self, profile: RewriteProfile) -> list[list[int]]:
        assert self.tokenizer is not None
        sequences: list[list[int]] = []
        for phrase in profile.control_phrases:
            for variant in (phrase, f" {phrase}"):
                token_ids = self.tokenizer.encode(variant, add_special_tokens=False)
                if token_ids and token_ids not in sequences:
                    sequences.append(token_ids)
        return sequences

    def _apply_control_phrase_bias(
        self,
        logits: Any,
        state: dict[str, Any],
        generated_ids: list[int],
    ) -> Any:
        phrase_sequences: list[list[int]] = state["phrase_sequences"]
        if not phrase_sequences:
            return logits
        if any(self._sequence_seen(generated_ids, sequence) for sequence in phrase_sequences):
            return logits

        bias = state["phrase_bias"]
        if state.get("force_control_phrase"):
            forced = self.torch.full_like(logits, -1.0e9)
            for sequence in phrase_sequences:
                prefix_len = self._matching_suffix_len(generated_ids, sequence)
                next_index = min(prefix_len, len(sequence) - 1)
                token_id = sequence[next_index]
                continuation_bonus = 4.0 if prefix_len else 0.0
                forced[:, token_id] = logits[:, token_id] + bias + continuation_bonus
            return forced

        biased = logits.clone()
        for sequence in phrase_sequences:
            if self._sequence_seen(generated_ids, sequence):
                continue
            prefix_len = self._matching_suffix_len(generated_ids, sequence)
            next_index = min(prefix_len, len(sequence) - 1)
            token_id = sequence[next_index]
            continuation_bonus = 2.0 if prefix_len else 0.0
            biased[:, token_id] = biased[:, token_id] + bias + continuation_bonus
        return biased

    def _control_phrase_complete(
        self,
        state: dict[str, Any],
        generated_ids: list[int],
        generated_text: str,
    ) -> bool:
        phrase_sequences: list[list[int]] = state["phrase_sequences"]
        if not any(self._sequence_seen(generated_ids, sequence) for sequence in phrase_sequences):
            return False
        return generated_text.strip().endswith((".", "!", "?"))

    @staticmethod
    def _sequence_seen(generated_ids: list[int], sequence: list[int]) -> bool:
        if len(sequence) > len(generated_ids):
            return False
        window = len(sequence)
        return any(
            generated_ids[index : index + window] == sequence
            for index in range(len(generated_ids) - window + 1)
        )

    @staticmethod
    def _matching_suffix_len(generated_ids: list[int], sequence: list[int]) -> int:
        max_len = min(len(generated_ids), len(sequence) - 1)
        for length in range(max_len, 0, -1):
            if generated_ids[-length:] == sequence[:length]:
                return length
        return 0

    def _build_vector_bank(self) -> _VectorBank:
        assert self.model is not None and self.tokenizer is not None and self.torch is not None
        directions: dict[str, Any] = {}
        jacobian_directions: dict[str, Any] | None = {} if self.jacobian_lens is not None else None
        baseline = self._activation_for_text(
            "This is an ordinary neutral enterprise assistant request with no hidden topic."
        )
        jacobian_baseline = self._transport_jacobian_residual(baseline) if jacobian_directions is not None else None
        for concept_id, concept in CONCEPTS.items():
            activations = [self._activation_for_text(prompt) for prompt in concept.calibration_prompts]
            mean_activation = self.torch.stack(activations).mean(dim=0)
            direction = mean_activation - baseline
            directions[concept_id] = direction / (direction.norm() + 1e-6)
            if jacobian_directions is not None and jacobian_baseline is not None:
                jacobian_activation = self._transport_jacobian_residual(mean_activation)
                jacobian_direction = jacobian_activation - jacobian_baseline
                jacobian_directions[concept_id] = jacobian_direction / (jacobian_direction.norm() + 1e-6)
        for profile in REWRITE_PROFILES.values():
            activations = [self._activation_for_text(prompt) for prompt in profile.target_prompts]
            mean_activation = self.torch.stack(activations).mean(dim=0)
            direction = mean_activation - baseline
            directions[profile.target_concept] = direction / (direction.norm() + 1e-6)
            if jacobian_directions is not None and jacobian_baseline is not None:
                jacobian_activation = self._transport_jacobian_residual(mean_activation)
                jacobian_direction = jacobian_activation - jacobian_baseline
                jacobian_directions[profile.target_concept] = jacobian_direction / (jacobian_direction.norm() + 1e-6)
        return _VectorBank(
            directions=directions,
            target_layer=self.layer,
            jacobian_directions=jacobian_directions,
        )

    def _activation_for_text(self, text: str) -> Any:
        assert self.model is not None
        input_ids = self._encode_chat(text)
        outputs = self._forward(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[self.layer + 1][0, -4:, :].mean(dim=0).detach()
        return hidden.float()

    def _build_custom_lens_bank(
        self,
        concepts: list[CustomConceptInput],
        lookup_injections: list[JSpaceInjection] | None = None,
    ) -> list[_CustomConceptVector]:
        assert self.torch is not None
        baseline = self._activation_for_text("Neutral enterprise assistant text with no user-defined concept.")
        jacobian_baseline = self._transport_jacobian_residual(baseline) if self.jacobian_lens is not None else None
        bank: list[_CustomConceptVector] = []
        for index, concept in enumerate(concepts):
            label = concept.label.strip()
            keywords = self._clean_custom_keywords(concept.keywords)
            if not keywords:
                raise ValueError(f"Custom concept {label!r} needs at least one non-empty keyword.")
            calibration_prompts = self._custom_calibration_prompts(label, keywords, lookup_injections)
            activations = [self._activation_for_text(prompt) for prompt in calibration_prompts]
            mean_activation = self.torch.stack(activations).mean(dim=0)
            proxy_direction = mean_activation - baseline
            proxy_direction = proxy_direction / (proxy_direction.norm() + 1e-6)
            jacobian_direction = None
            if jacobian_baseline is not None:
                jacobian_activation = self._transport_jacobian_residual(mean_activation)
                jacobian_direction = jacobian_activation - jacobian_baseline
                jacobian_direction = jacobian_direction / (jacobian_direction.norm() + 1e-6)
            bank.append(
                _CustomConceptVector(
                    id=self._custom_concept_id(index, label),
                    label=label,
                    polarity=concept.polarity,
                    keywords=keywords,
                    calibration_prompts=calibration_prompts,
                    proxy_direction=proxy_direction,
                    jacobian_direction=jacobian_direction,
                )
            )
        return bank

    @staticmethod
    def _custom_concept_id(index: int, label: str) -> str:
        slug = []
        previous_separator = False
        for char in label.lower():
            if char.isalnum():
                slug.append(char)
                previous_separator = False
            elif not previous_separator:
                slug.append("_")
                previous_separator = True
        cleaned = "".join(slug).strip("_") or "concept"
        return f"custom_{index}_{cleaned[:40]}"

    @staticmethod
    def _clean_custom_keywords(keywords: list[str]) -> list[str]:
        cleaned: list[str] = []
        for keyword in keywords:
            value = keyword.strip()
            if value and value.lower() not in {item.lower() for item in cleaned}:
                cleaned.append(value)
        return cleaned[:12]

    @staticmethod
    def _custom_calibration_prompts(
        label: str,
        keywords: list[str],
        lookup_injections: list[JSpaceInjection] | None = None,
    ) -> list[str]:
        joined = ", ".join(keywords)
        prompts = [
            f"The hidden concept is {label}. Related terms: {joined}.",
            f"The assistant is thinking about {label}: {joined}.",
            f"A model trajectory focused on {label}, especially {joined}.",
        ]
        prompts.extend(
            injection.injected_text
            for injection in lookup_injections or []
            if HFJSpaceEngine._injection_targets_custom_concept(injection, label, keywords)
        )
        return prompts

    def _lookup_injection_direction(self, lookup_injections: list[JSpaceInjection]) -> Any | None:
        if not lookup_injections:
            return None
        assert self.torch is not None
        baseline = self._activation_for_text("Neutral enterprise assistant text with no retrieved lookup context.")
        activations = [self._activation_for_text(injection.injected_text) for injection in lookup_injections[:3]]
        mean_activation = self.torch.stack(activations).mean(dim=0)
        direction = mean_activation - baseline
        return direction / (direction.norm() + 1e-6)

    @staticmethod
    def _injection_targets_custom_concept(
        injection: JSpaceInjection,
        label: str,
        keywords: list[str],
    ) -> bool:
        targets = {target.casefold() for target in injection.concept_targets}
        if label.strip().casefold() in targets:
            return True
        lookup_text = f"{injection.keyword} {injection.title} {injection.summary}".casefold()
        return any(keyword.strip().casefold() in lookup_text for keyword in keywords if keyword.strip())

    def _score_custom_proxy_sequence(
        self,
        hidden_states: Any,
        bank: list[_CustomConceptVector],
    ) -> list[ConceptScore]:
        assert self.torch is not None
        vectors = hidden_states.detach().float()
        vectors = vectors / (vectors.norm(dim=-1, keepdim=True) + 1e-6)
        scores: list[ConceptScore] = []
        for concept in bank:
            direction = concept.proxy_direction.to(vectors.device).float()
            cosine = vectors @ direction
            row_scores = ((cosine + 0.25) / 0.75).clamp(0.0, 1.0)
            best_row = int(row_scores.argmax().item())
            score = float(row_scores[best_row].detach().cpu())
            logits = self._proxy_logits(hidden_states[best_row])
            scores.append(
                ConceptScore(
                    id=concept.id,
                    label=concept.label,
                    score=round(score, 3),
                    polarity=concept.polarity,
                    evidence=self._top_keyword_readout_tokens(logits, concept.keywords),
                )
            )
        return sorted(scores, key=lambda item: item.score, reverse=True)[:5]

    def _score_custom_jacobian_sequence(
        self,
        hidden_states: Any,
        bank: list[_CustomConceptVector],
    ) -> tuple[list[ConceptScore], list[str]]:
        if self.jacobian_lens is None:
            return [], []
        assert self.torch is not None
        transported = self._transport_jacobian_residual(hidden_states)
        vectors = transported.detach().float()
        vectors = vectors / (vectors.norm(dim=-1, keepdim=True) + 1e-6)
        scores: list[ConceptScore] = []
        best_overall = -1.0
        token_row = 0
        for concept in bank:
            if concept.jacobian_direction is None:
                continue
            direction = concept.jacobian_direction.to(vectors.device).float()
            cosine = vectors @ direction
            row_scores = self.torch.sigmoid((cosine + 0.05) * 6.0).clamp(0.0, 1.0)
            best_row = int(row_scores.argmax().item())
            score = float(row_scores[best_row].detach().cpu())
            if score > best_overall:
                best_overall = score
                token_row = best_row
            logits = self._jacobian_logits(transported[best_row])
            scores.append(
                ConceptScore(
                    id=concept.id,
                    label=concept.label,
                    score=round(score, 3),
                    polarity=concept.polarity,
                    evidence=self._top_keyword_readout_tokens(logits, concept.keywords),
                )
            )
        scores = sorted(scores, key=lambda item: item.score, reverse=True)[:5]
        return scores, self._jacobian_summary_tokens(scores, self._jacobian_logits(transported[token_row]))

    def _proxy_logits(self, hidden: Any) -> Any:
        assert self.model is not None and self.torch is not None
        with self.torch.no_grad():
            projected = hidden.reshape(1, 1, -1).to(next(self.model.parameters()).device)
            if hasattr(self.model.model, "norm"):
                projected = self.model.model.norm(projected)
            return self.model.lm_head(projected)[0, 0].detach().float().cpu()

    def _top_keyword_readout_tokens(self, logits: Any, keywords: list[str]) -> list[str]:
        assert self.tokenizer is not None
        token_ids = self._keyword_token_ids_from_keywords(keywords)
        if not token_ids:
            return keywords[:3]
        selected = sorted(
            (
                (float(logits[token_id].detach().cpu()), token_id)
                for token_id in set(token_ids)
                if 0 <= token_id < int(logits.numel())
            ),
            reverse=True,
        )[:3]
        evidence = [self.tokenizer.decode([token_id], skip_special_tokens=True).strip() for _, token_id in selected]
        return [token for token in evidence if token] or keywords[:3]

    def _keyword_token_ids_from_keywords(self, keywords: list[str]) -> list[int]:
        assert self.tokenizer is not None
        token_ids: list[int] = []
        for keyword in keywords:
            for variant in (keyword, f" {keyword}"):
                encoded = self.tokenizer.encode(variant, add_special_tokens=False)
                if encoded:
                    token_ids.append(int(encoded[0]))
        return list(dict.fromkeys(token_ids))

    @staticmethod
    def _custom_lens_summary(
        top_proxy: ConceptScore | None,
        top_jacobian: ConceptScore | None,
        alignment: str,
        concept_count: int,
        injection_count: int = 0,
    ) -> str:
        injection_clause = (
            f" {injection_count} keyword lookup{'s' if injection_count != 1 else ''} injected into J-space calibration."
            if injection_count
            else ""
        )
        if top_proxy is None:
            return (
                f"Generated {concept_count} custom concept directions; no prompt activation was scored."
                f"{injection_clause}"
            )
        if top_jacobian is None:
            return (
                f"Generated {concept_count} custom concept directions; proxy lens top concept is "
                f"{top_proxy.label}.{injection_clause}"
            )
        return (
            f"Generated {concept_count} custom concept directions; proxy top is {top_proxy.label}, "
            f"Jacobian top is {top_jacobian.label}, and the comparison {alignment}.{injection_clause}"
        )

    @staticmethod
    def _custom_lens_note(injections: list[JSpaceInjection]) -> str:
        if not injections:
            return "User-defined concept lens scored in raw residual space and Jacobian-transported J-space."
        return (
            "Keyword lookup pulled additional context and injected it into the custom concept "
            "calibration path before raw residual and Jacobian-transported scoring."
        )

    def _score_readouts(
        self,
        hidden: Any,
        concept_ids: list[str],
    ) -> tuple[list[ConceptScore], list[ConceptScore], list[ConceptScore], list[str]]:
        proxy_scores = self._score_hidden(hidden, concept_ids)
        jacobian_scores, jacobian_tokens = self._score_jacobian_hidden(hidden, concept_ids)
        control_scores = proxy_scores
        return control_scores, proxy_scores, jacobian_scores, jacobian_tokens

    def _score_prompt_readouts(
        self,
        hidden_states: Any,
        concept_ids: list[str],
    ) -> tuple[list[ConceptScore], list[ConceptScore], list[ConceptScore], list[str]]:
        tail_count = min(64, int(hidden_states.shape[0]))
        tail = hidden_states[-tail_count:]
        proxy_hidden = tail[-min(4, tail_count) :].mean(dim=0)
        proxy_scores = self._score_hidden(proxy_hidden, concept_ids)
        jacobian_scores, jacobian_tokens = self._score_jacobian_sequence(tail, concept_ids)
        control_scores = proxy_scores
        return control_scores, proxy_scores, jacobian_scores, jacobian_tokens

    def _layer_activation_scan(
        self,
        hidden_states: Any,
        concept_ids: list[str],
        prompt_window: int = 1,
    ) -> list[LayerActivation]:
        assert self.bank is not None and self.torch is not None
        rows: list[LayerActivation] = []
        for layer_index, layer_hidden in enumerate(hidden_states[1:]):
            hidden = self._layer_vector(layer_hidden, prompt_window)
            activation = self._top_layer_activation(hidden, concept_ids)
            if activation is not None:
                rows.append(activation.model_copy(update={"layer": layer_index}))
        return rows

    def _jlens_layer_readouts(
        self,
        hidden_states: Any,
        concept_ids: list[str],
        rank_threshold: int,
        policy_terms: list[str] | None = None,
    ) -> list[LayerActivation]:
        """Decode every fitted layer and mark policy words by vocabulary rank."""
        if self.jacobian_lens is None or self.torch is None:
            raise RuntimeError("Anthropic Jacobian lens is required for monitoring but is unavailable.")

        policy_tokens = self._lens_policy_token_lookup(concept_ids, policy_terms)
        residuals = self.torch.stack(
            [
                self._layer_vector(hidden_states[layer + 1], prompt_window=1)
                for layer in self.jacobian_lens.source_layers
            ]
        ).to(
            device=self.jacobian_lens.matrix_stack.device,
            dtype=self.jacobian_lens.matrix_stack.dtype,
        )
        with self.torch.no_grad():
            transported = self.torch.bmm(
                residuals.unsqueeze(1),
                self.jacobian_lens.matrix_stack.transpose(1, 2),
            ).squeeze(1)
            lens_logits = self._jacobian_logits(transported)
        top_count = min(max(rank_threshold, 8), int(lens_logits.shape[-1]))
        ranked_token_ids = self.torch.topk(lens_logits, k=top_count, dim=-1).indices.tolist()

        rows: list[LayerActivation] = []
        for layer, token_ids in zip(self.jacobian_lens.source_layers, ranked_token_ids, strict=True):
            top_tokens = self._decode_ranked_tokens(token_ids, limit=5)

            match: tuple[int, ConceptSpec, str, str] | None = None
            for rank, token_id in enumerate(token_ids, start=1):
                policy = policy_tokens.get(int(token_id))
                if policy is None:
                    continue
                concept, policy_term = policy
                decoded = self.tokenizer.decode([token_id], skip_special_tokens=True).strip() or policy_term
                match = (rank, concept, policy_term, decoded)
                break

            if match is None:
                rows.append(
                    LayerActivation(
                        layer=layer,
                        concept_id="jlens_readout",
                        label=top_tokens[0] if top_tokens else "No readable token",
                        score=0.0,
                        polarity="neutral",
                        top_tokens=top_tokens,
                    )
                )
                continue

            rank, concept, policy_term, decoded = match
            rows.append(
                LayerActivation(
                    layer=layer,
                    concept_id=concept.id,
                    label=concept.label,
                    score=round((rank_threshold - rank + 1) / rank_threshold, 3),
                    polarity=concept.polarity,
                    rank=rank,
                    matched_token=decoded,
                    top_tokens=list(dict.fromkeys([decoded, *top_tokens]))[:5],
                )
            )
        return rows

    def _lens_policy_token_lookup(
        self,
        concept_ids: list[str],
        policy_terms: list[str] | None = None,
    ) -> dict[int, tuple[ConceptSpec, str]]:
        lookup: dict[int, tuple[ConceptSpec, str]] = {}
        allowed_terms = {term.strip().casefold() for term in policy_terms or [] if term.strip()}
        assigned_terms: set[str] = set()
        for concept_id in concept_ids:
            concept = CONCEPTS.get(concept_id)
            if concept is None or concept.polarity != "risk":
                continue
            for token_id, term in self._single_token_policy_terms(concept).items():
                if policy_terms is not None and term.casefold() not in allowed_terms:
                    continue
                lookup.setdefault(token_id, (concept, term))
                assigned_terms.add(term.casefold())

        custom_terms = [term for term in policy_terms or [] if term.strip().casefold() not in assigned_terms]
        if custom_terms:
            custom_policy = ConceptSpec(
                id="custom_policy",
                label="Custom Policy Word",
                description="User-supplied single-token Qwen vocabulary filter.",
                polarity="risk",
                keywords=tuple(custom_terms),
                calibration_prompts=(),
            )
            for term in custom_terms:
                for token_id in self._single_token_ids_for_term(term):
                    lookup.setdefault(token_id, (custom_policy, term.strip()))
        return lookup

    def _single_token_policy_terms(self, concept: ConceptSpec) -> dict[int, str]:
        assert self.tokenizer is not None
        cached = self._lens_filter_token_cache.get(concept.id)
        if cached is not None:
            return cached

        token_ids: dict[int, str] = {}
        for keyword in concept.keywords:
            term = keyword.strip()
            if not term:
                continue
            variants = (term, f" {term}", term.capitalize(), f" {term.capitalize()}", term.upper(), f" {term.upper()}")
            for variant in variants:
                encoded = self.tokenizer.encode(variant, add_special_tokens=False)
                if len(encoded) == 1:
                    token_ids.setdefault(int(encoded[0]), term)
        self._lens_filter_token_cache[concept.id] = token_ids
        return token_ids

    def _decode_ranked_tokens(self, token_ids: list[int], limit: int) -> list[str]:
        assert self.tokenizer is not None
        tokens: list[str] = []
        for token_id in token_ids:
            token = self.tokenizer.decode([token_id], skip_special_tokens=True).strip()
            if not token or token in tokens:
                continue
            tokens.append(token)
            if len(tokens) >= limit:
                break
        return tokens

    @staticmethod
    def _lens_match_scores(layer_readouts: list[LayerActivation]) -> list[ConceptScore]:
        strongest: dict[str, tuple[LayerActivation, list[str]]] = {}
        for readout in layer_readouts:
            if readout.rank is None or readout.polarity != "risk":
                continue
            current = strongest.get(readout.concept_id)
            evidence = [readout.matched_token or readout.label, f"L{readout.layer} rank {readout.rank}"]
            if current is None or readout.rank < (current[0].rank or 10**9):
                strongest[readout.concept_id] = (readout, evidence)
        return sorted(
            [
                ConceptScore(
                    id=readout.concept_id,
                    label=readout.label,
                    score=readout.score,
                    polarity=readout.polarity,
                    evidence=evidence,
                )
                for readout, evidence in strongest.values()
            ],
            key=lambda item: item.score,
            reverse=True,
        )

    @staticmethod
    def _best_ranked_readout(readouts: list[LayerActivation]) -> LayerActivation | None:
        matches = [readout for readout in readouts if readout.rank is not None]
        return min(matches, key=lambda item: item.rank or 10**9, default=None)

    @classmethod
    def _best_lens_match(cls, frame: TraceFrame) -> LayerActivation | None:
        return cls._best_ranked_readout(frame.layer_activations)

    @classmethod
    def _monitor_completion_summary(cls, frames: list[TraceFrame]) -> str:
        matches = sum(
            1
            for frame in frames
            for readout in frame.layer_activations
            if readout.polarity == "risk" and readout.rank is not None
        )
        if matches:
            return f"Observed {matches} unsafe Anthropic J-lens rank matches; generation was allowed to continue."
        return (
            "Run completed with direct Anthropic J-lens vocabulary monitoring; "
            "no unsafe word crossed the rank cutoff."
        )

    def _custom_layer_activation_scan(
        self,
        hidden_states: Any,
        bank: list[_CustomConceptVector],
        prompt_window: int = 1,
    ) -> list[LayerActivation]:
        rows: list[LayerActivation] = []
        for layer_index, layer_hidden in enumerate(hidden_states[1:]):
            hidden = self._layer_vector(layer_hidden, prompt_window)
            activation = self._top_custom_layer_activation(hidden, bank)
            if activation is not None:
                rows.append(activation.model_copy(update={"layer": layer_index}))
        return rows

    def _layer_vector(self, layer_hidden: Any, prompt_window: int) -> Any:
        if layer_hidden.ndim == 3:
            layer_hidden = layer_hidden[0]
        if layer_hidden.ndim == 2:
            window = max(1, min(prompt_window, int(layer_hidden.shape[0])))
            return layer_hidden[-window:].mean(dim=0).detach()
        return layer_hidden.detach()

    def _top_layer_activation(self, hidden: Any, concept_ids: list[str]) -> LayerActivation | None:
        assert self.bank is not None
        hidden_f = hidden.detach().float()
        hidden_f = hidden_f / (hidden_f.norm() + 1e-6)
        best: LayerActivation | None = None
        for concept_id in concept_ids:
            direction = self.bank.directions.get(concept_id)
            concept = CONCEPTS.get(concept_id)
            if direction is None or concept is None:
                continue
            score = self._direction_score(hidden_f, direction.to(hidden_f.device))
            candidate = LayerActivation(
                layer=0,
                concept_id=concept.id,
                label=concept.label,
                score=round(score, 3),
                polarity=concept.polarity,
            )
            if best is None or candidate.score > best.score:
                best = candidate
        return best

    def _top_custom_layer_activation(
        self,
        hidden: Any,
        bank: list[_CustomConceptVector],
    ) -> LayerActivation | None:
        hidden_f = hidden.detach().float()
        hidden_f = hidden_f / (hidden_f.norm() + 1e-6)
        best: LayerActivation | None = None
        for concept in bank:
            score = self._direction_score(hidden_f, concept.proxy_direction.to(hidden_f.device))
            candidate = LayerActivation(
                layer=0,
                concept_id=concept.id,
                label=concept.label,
                score=round(score, 3),
                polarity=concept.polarity,
            )
            if best is None or candidate.score > best.score:
                best = candidate
        return best

    @staticmethod
    def _direction_score(hidden: Any, direction: Any) -> float:
        cosine = float((hidden * direction).sum().cpu())
        return max(0.0, min(1.0, (cosine + 0.25) / 0.75))

    def _score_hidden(self, hidden: Any, concept_ids: list[str]) -> list[ConceptScore]:
        assert self.bank is not None and self.torch is not None
        scores: list[ConceptScore] = []
        hidden_f = hidden.detach().float()
        hidden_f = hidden_f / (hidden_f.norm() + 1e-6)
        for concept_id in concept_ids:
            concept = CONCEPTS[concept_id]
            direction = self.bank.directions[concept_id].to(hidden_f.device)
            score = self._direction_score(hidden_f, direction)
            evidence = self._top_readout_tokens(hidden, concept)
            scores.append(
                ConceptScore(
                    id=concept.id,
                    label=concept.label,
                    score=round(score, 3),
                    polarity=concept.polarity,
                    evidence=evidence,
                )
            )
        return sorted(scores, key=lambda item: item.score, reverse=True)[:5]

    def _transport_jacobian_residual(self, residual: Any, layer: int | None = None) -> Any:
        assert self.jacobian_lens is not None and self.torch is not None
        original_shape = tuple(residual.shape)
        source_layer = self.layer if layer is None else layer
        matrix = self.jacobian_lens.matrices[source_layer]
        with self.torch.no_grad():
            residual_batch = residual.detach().reshape(-1, original_shape[-1]).to(
                device=matrix.device,
                dtype=matrix.dtype,
            )
            transported = residual_batch @ matrix.T
        if len(original_shape) == 1:
            return transported[0].float()
        return transported.reshape(*original_shape[:-1], transported.shape[-1]).float()

    def _jacobian_logits(self, transported: Any) -> Any:
        assert self.jacobian_lens is not None and self.torch is not None
        matrix = self.jacobian_lens.matrices[self.layer]
        with self.torch.no_grad():
            residual_batch = transported.detach().reshape(-1, transported.shape[-1]).to(
                device=matrix.device,
                dtype=matrix.dtype,
            )
            logits = self.jacobian_lens.lens_model.unembed(residual_batch).detach().float().cpu()
        return logits[0] if logits.shape[0] == 1 else logits

    def _score_jacobian_hidden(self, hidden: Any, concept_ids: list[str]) -> tuple[list[ConceptScore], list[str]]:
        if self.jacobian_lens is None or self.bank is None or self.bank.jacobian_directions is None:
            return [], []
        try:
            transported = self._transport_jacobian_residual(hidden)
            scores, token_row = self._score_jacobian_transported_sequence(transported.reshape(1, -1), concept_ids)
            logits = self._jacobian_logits(transported.reshape(1, -1)[token_row])
            return scores, self._jacobian_summary_tokens(scores, logits)
        except Exception as exc:
            self.jacobian_lens_error = f"Jacobian lens readout failed: {exc}"
            return [], []

    def _score_jacobian_sequence(
        self,
        hidden_states: Any,
        concept_ids: list[str],
    ) -> tuple[list[ConceptScore], list[str]]:
        if self.jacobian_lens is None or self.bank is None or self.bank.jacobian_directions is None:
            return [], []
        try:
            transported = self._transport_jacobian_residual(hidden_states)
            scores, token_row = self._score_jacobian_transported_sequence(transported, concept_ids)
            logits = self._jacobian_logits(transported[token_row])
            return scores, self._jacobian_summary_tokens(scores, logits)
        except Exception as exc:
            self.jacobian_lens_error = f"Jacobian lens readout failed: {exc}"
            return [], []

    def _score_jacobian_transported_sequence(
        self,
        transported: Any,
        concept_ids: list[str],
    ) -> tuple[list[ConceptScore], int]:
        assert self.bank is not None and self.bank.jacobian_directions is not None and self.torch is not None
        if transported.ndim == 1:
            transported = transported.reshape(1, -1)
        vectors = transported.detach().float()
        vectors = vectors / (vectors.norm(dim=-1, keepdim=True) + 1e-6)

        scores: list[ConceptScore] = []
        token_row = 0
        best_overall = -1.0
        for concept_id in concept_ids:
            concept = CONCEPTS[concept_id]
            direction = self.bank.jacobian_directions[concept_id].to(vectors.device).float()
            cosine = vectors @ direction
            row_scores = self.torch.sigmoid((cosine + 0.05) * 6.0).clamp(0.0, 1.0)
            best_row = int(row_scores.argmax().item())
            score = float(row_scores[best_row].detach().cpu())
            if score > best_overall:
                best_overall = score
                token_row = best_row

            scores.append(
                ConceptScore(
                    id=concept.id,
                    label=concept.label,
                    score=round(score, 3),
                    polarity=concept.polarity,
                    evidence=self._top_jacobian_readout_tokens(transported[best_row], concept),
                )
            )
        return sorted(scores, key=lambda item: item.score, reverse=True)[:5], token_row

    def _top_jacobian_readout_tokens(self, transported: Any, concept: ConceptSpec) -> list[str]:
        assert self.tokenizer is not None
        try:
            logits = self._jacobian_logits(transported)
            token_ids = self._keyword_token_ids(concept)
            if not token_ids:
                return list(concept.keywords[:2])
            selected = sorted(
                (
                    (float(logits[token_id].detach().cpu()), token_id)
                    for token_id in set(token_ids)
                    if 0 <= token_id < int(logits.numel())
                ),
                reverse=True,
            )[:3]
            evidence = [
                self.tokenizer.decode([token_id], skip_special_tokens=True).strip()
                for _, token_id in selected
            ]
            return [token for token in evidence if token] or list(concept.keywords[:2])
        except Exception:
            return list(concept.keywords[:2])

    def _jacobian_summary_tokens(self, scores: list[ConceptScore], logits: Any) -> list[str]:
        tokens: list[str] = []
        for score in scores:
            for token in score.evidence:
                if token and token not in tokens:
                    tokens.append(token)
                if len(tokens) >= 8:
                    return tokens
        for token in self._decode_top_tokens(logits):
            if token and token not in tokens:
                tokens.append(token)
            if len(tokens) >= 8:
                break
        return tokens

    def _decode_top_tokens(self, logits: Any, limit: int = 8) -> list[str]:
        assert self.tokenizer is not None and self.torch is not None
        candidate_count = min(limit * 4, int(logits.numel()))
        token_ids = self.torch.topk(logits, candidate_count).indices.tolist()
        tokens: list[str] = []
        for token_id in token_ids:
            token = self.tokenizer.decode([int(token_id)], skip_special_tokens=True).strip()
            if token and token not in tokens:
                tokens.append(token)
            if len(tokens) >= limit:
                break
        return tokens

    def _top_readout_tokens(self, hidden: Any, concept: ConceptSpec) -> list[str]:
        assert self.model is not None and self.tokenizer is not None and self.torch is not None
        try:
            with self.torch.no_grad():
                projected = hidden.reshape(1, 1, -1).to(next(self.model.parameters()).device)
                if hasattr(self.model.model, "norm"):
                    projected = self.model.model.norm(projected)
                logits = self.model.lm_head(projected)[0, 0]
                token_ids = self._keyword_token_ids(concept)
                if not token_ids:
                    return list(concept.keywords[:2])
                selected = sorted(
                    ((float(logits[token_id].detach().cpu()), token_id) for token_id in set(token_ids)),
                    reverse=True,
                )[:3]
                return [
                    self.tokenizer.decode([token_id], skip_special_tokens=True).strip()
                    for _, token_id in selected
                ]
        except Exception:
            return list(concept.keywords[:2])

    def _keyword_token_ids(self, concept: ConceptSpec) -> list[int]:
        assert self.tokenizer is not None
        cached = self._concept_token_cache.get(concept.id)
        if cached is not None:
            return cached
        token_ids: list[int] = []
        for keyword in concept.keywords:
            for variant in (keyword, f" {keyword}"):
                encoded = self.tokenizer.encode(variant, add_special_tokens=False)
                if encoded:
                    token_ids.append(int(encoded[0]))
        deduped = list(dict.fromkeys(token_ids))
        self._concept_token_cache[concept.id] = deduped
        return deduped

    def _encode_chat(self, prompt: str) -> Any:
        assert self.model is not None and self.tokenizer is not None and self.torch is not None
        messages = [{"role": "user", "content": prompt}]
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            )
        except TypeError:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        input_ids = self._extract_input_ids(encoded)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        return input_ids.to(next(self.model.parameters()).device)

    def _extract_input_ids(self, encoded: Any) -> Any:
        assert self.torch is not None
        input_ids = encoded
        for _ in range(5):
            if self.torch.is_tensor(input_ids):
                return input_ids
            if isinstance(input_ids, dict) and "input_ids" in input_ids:
                input_ids = input_ids["input_ids"]
                continue
            if hasattr(input_ids, "input_ids"):
                input_ids = input_ids.input_ids
                continue
            if isinstance(input_ids, list):
                return self.torch.tensor(input_ids)
            break
        raise TypeError(f"Tokenizer returned unsupported input_ids type: {type(input_ids)!r}")

    def _forward(self, input_ids: Any, output_hidden_states: bool = False) -> Any:
        assert self.model is not None and self.torch is not None
        with self.torch.no_grad():
            return self.model(input_ids=input_ids, output_hidden_states=output_hidden_states)

    def _sample(self, logits: Any, temperature: float, top_p: float, seed: int) -> Any:
        assert self.torch is not None
        torch = self.torch
        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1)
        generator = torch.Generator(device=logits.device)
        generator.manual_seed(seed + random.randint(0, 1_000_000))
        probs = torch.softmax(logits / temperature, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= top_p
        keep[..., 0] = True
        filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        filtered = filtered / filtered.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(filtered, num_samples=1, generator=generator)
        return sorted_indices.gather(-1, sampled).squeeze(-1)

    @staticmethod
    def _frame_peak(frame: TraceFrame) -> float:
        return max((score.score for score in frame.top_concepts), default=0.0)
