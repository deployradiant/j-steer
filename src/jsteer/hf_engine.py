from __future__ import annotations

import os
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .concepts import CONCEPTS, REWRITE_PROFILES, ConceptSpec, RewriteProfile
from .schemas import (
    ConceptScore,
    EngineInfo,
    Intervention,
    JacobianLensInfo,
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
    lens_model: Any
    matrix: Any
    source: str
    artifact: str
    layer: int
    d_model: int
    n_prompts: int


class HFJSpaceEngine:
    """Qwen hidden-state monitor and activation rewrite engine.

    The control path uses calibrated concept directions because those directions are editable.
    In parallel, the readout path decodes the same residual through Anthropic's published
    Jacobian-lens implementation when a compatible Qwen lens artifact is configured.
    """

    mode = "hf"
    readout = "concept-vector control + Jacobian lens readout comparison"

    def __init__(self) -> None:
        self.model_name = os.getenv("JSTEER_MODEL", "Qwen/Qwen3-1.7B")
        self.lens_source = os.getenv("JSTEER_JLENS_REPO", "neuronpedia/jacobian-lens")
        self.lens_artifact = os.getenv(
            "JSTEER_JLENS_FILE",
            "qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt",
        )
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
        self._ensure_loaded()
        started = time.perf_counter()
        if request.mode == "rewrite":
            response = self._run_rewrite(request)
        else:
            response = self._run_monitor(request)
        response.metrics.latency_ms = int((time.perf_counter() - started) * 1000)
        return response

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
        self.bank = self._build_vector_bank()
        self._loaded = True

    def _jacobian_lens_info(self) -> JacobianLensInfo:
        if self.jacobian_lens is not None:
            return JacobianLensInfo(
                available=True,
                status="ready",
                source=self.jacobian_lens.source,
                artifact=self.jacobian_lens.artifact,
                layer=self.jacobian_lens.layer,
                message=(
                    f"Loaded d_model={self.jacobian_lens.d_model} lens from "
                    f"{self.jacobian_lens.n_prompts} prompts."
                ),
            )
        if self.jacobian_lens_error:
            return JacobianLensInfo(
                available=False,
                status="unavailable",
                source=self.lens_source,
                artifact=self.lens_artifact,
                layer=max(self.layer, 0),
                message=self.jacobian_lens_error,
            )
        return JacobianLensInfo(
            available=False,
            status="configured",
            source=self.lens_source,
            artifact=self.lens_artifact,
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
            matrix = lens.jacobians[self.layer].to(device=parameter.device, dtype=parameter.dtype)
            self.jacobian_lens = _JacobianLensRuntime(
                lens_model=lens_model,
                matrix=matrix,
                source=self.lens_source,
                artifact=self.lens_artifact,
                layer=self.layer,
                d_model=int(lens.d_model),
                n_prompts=int(lens.n_prompts),
            )
            self.jacobian_lens_error = None
        except Exception as exc:
            self.jacobian_lens_error = f"Unable to load Jacobian lens: {exc}"

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
        prompt_hidden_states = outputs.hidden_states[self.layer + 1][0]
        prompt_scores, prompt_proxy, prompt_jacobian, prompt_jacobian_tokens = self._score_prompt_readouts(
            prompt_hidden_states,
            monitor_ids,
        )
        peak_score = max((score.score for score in prompt_scores), default=0.0)
        frames = [
            TraceFrame(
                step=0,
                phase="prompt",
                layer=self.layer,
                token="<prompt>",
                top_concepts=prompt_scores,
                proxy_concepts=prompt_proxy,
                jacobian_concepts=prompt_jacobian,
                jacobian_tokens=prompt_jacobian_tokens,
                action="observe",
                note="Prompt boundary hidden state before the first decoded token.",
            )
        ]
        if request.action == "kill" and peak_score >= request.threshold:
            frames[0].action = "blocked"
            frames[0].note = "Tripwire crossed before decoding."
            return RunResponse(
                id=str(uuid.uuid4()),
                status="blocked",
                mode="monitor",
                prompt=request.prompt,
                output="",
                summary=(
                    f"Run killed before output: {prompt_scores[0].label} scored "
                    f"{prompt_scores[0].score:.2f}."
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
                frame_peak = self._frame_peak(frame)
                if frame_peak >= request.threshold:
                    frame.action = "blocked"
                    frame.note = "Tripwire crossed during hidden-state decode monitoring."
                    blocked_frames = frames + decode_frames[: index + 1]
                    top = frame.top_concepts[0]
                    return RunResponse(
                        id=str(uuid.uuid4()),
                        status="blocked",
                        mode="monitor",
                        prompt=request.prompt,
                        output="",
                        summary=(
                            f"Run killed during decode: {top.label} reached "
                            f"{top.score:.2f} at step {frame.step}."
                        ),
                        engine=self.info(),
                        frames=blocked_frames,
                        metrics=RunMetrics(
                            prompt_tokens=int(input_ids.shape[-1]),
                            completion_tokens=0,
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
            summary="Run completed with J-space proxy monitoring enabled.",
            engine=self.info(),
            frames=frames,
            metrics=RunMetrics(
                prompt_tokens=int(input_ids.shape[-1]),
                completion_tokens=len(self.tokenizer.encode(text, add_special_tokens=False)),
                peak_score=max([peak_score, *[self._frame_peak(frame) for frame in decode_frames]]),
            ),
        )

    def _run_rewrite(self, request: RunRequest) -> RunResponse:
        assert self.model is not None and self.tokenizer is not None
        profile = REWRITE_PROFILES.get(request.rewrite_profile_id or "", REWRITE_PROFILES["brand_canon_injection"])
        input_ids = self._encode_chat(request.prompt)
        monitor_ids = list(profile.suppress_concepts) + [profile.target_concept]

        baseline_text = None
        baseline_frames: list[TraceFrame] = []
        if request.compare_baseline:
            baseline_text, baseline_frames = self._generate(
                input_ids=input_ids,
                request=request,
                monitor_ids=monitor_ids,
                steering=None,
            )

        steering = self._steering_state(profile, request)
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
            summary=f"{len(interventions)} forward-hook rewrites applied at layer {self.layer}.",
            engine=self.info(),
            frames=frames,
            baseline_frames=baseline_frames,
            interventions=interventions,
            metrics=RunMetrics(
                prompt_tokens=int(input_ids.shape[-1]),
                completion_tokens=len(self.tokenizer.encode(steered_text, add_special_tokens=False)),
                peak_score=peak,
                rewrites=len(interventions),
            ),
        )

    def _generate(
        self,
        input_ids: Any,
        request: RunRequest,
        monitor_ids: list[str],
        steering: dict[str, Any] | None,
    ) -> tuple[str, list[TraceFrame]]:
        assert self.model is not None and self.tokenizer is not None and self.torch is not None
        torch = self.torch
        frames: list[TraceFrame] = []
        handle = None
        state = steering
        if state is not None:
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
                hidden = outputs.hidden_states[self.layer + 1][0, -1, :]
                scores, proxy_scores, jacobian_scores, jacobian_tokens = self._score_readouts(hidden, monitor_ids)
                logits = outputs.logits[:, -1, :]
                if state is not None:
                    logits = self._apply_control_phrase_bias(logits, state, generated_ids)
                next_token = self._sample(logits, request.temperature, request.top_p, request.seed + step)
                token_id = int(next_token.item())
                if token_id == self.tokenizer.eos_token_id:
                    break
                generated_ids.append(token_id)
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                action = "rewrite" if state and state.pop("last_rewrite", False) else "observe"
                frames.append(
                    TraceFrame(
                        step=step + 1,
                        phase="decode",
                        layer=self.layer,
                        token=self.tokenizer.decode([token_id], skip_special_tokens=True),
                        generated_text=generated_text,
                        top_concepts=scores,
                        proxy_concepts=proxy_scores,
                        jacobian_concepts=jacobian_scores,
                        jacobian_tokens=jacobian_tokens,
                        action=action,
                        note=(
                            "Forward hook edited hidden state; decoder readout favored approved terms."
                            if action == "rewrite"
                            else None
                        ),
                    )
                )
                if state is not None and self._control_phrase_complete(state, generated_ids, generated_text):
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
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip(), frames
        finally:
            if handle is not None:
                handle.remove()

    def _steering_state(self, profile: RewriteProfile, request: RunRequest) -> dict[str, Any]:
        assert self.bank is not None and self.tokenizer is not None
        torch = self.torch
        target = self.bank.directions[profile.target_concept]
        suppressors = [self.bank.directions[concept_id] for concept_id in profile.suppress_concepts]
        suppress = torch.stack(suppressors).mean(dim=0) if suppressors else torch.zeros_like(target)
        delta = request.strength * target - (request.strength * 0.7) * suppress
        phrase_sequences = self._control_phrase_sequences(profile)
        return {
            "profile": profile,
            "delta": delta,
            "phrase_sequences": phrase_sequences,
            "phrase_bias": max(0.0, 4.0 + request.strength * 2.2),
            "force_control_phrase": True,
            "threshold": request.threshold,
            "monitor_ids": list(profile.suppress_concepts),
            "interventions": [],
            "last_rewrite": False,
            "step": 0,
        }

    def _steering_hook(self, output: Any, state: dict[str, Any]) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        last = hidden[0, -1, :].detach()
        scores = self._score_hidden(last, state["monitor_ids"])
        peak = max((score.score for score in scores), default=0.0)
        state["step"] += 1
        should_rewrite = peak >= state["threshold"] or state["step"] <= 3
        state["last_rewrite"] = should_rewrite
        if not should_rewrite:
            return output

        delta = state["delta"].to(device=hidden.device, dtype=hidden.dtype)
        edited = hidden.clone()
        edited[:, -1, :] = edited[:, -1, :] + delta
        profile: RewriteProfile = state["profile"]
        suppress_label = ", ".join(CONCEPTS[concept_id].label for concept_id in profile.suppress_concepts)
        state["interventions"].append(
            Intervention(
                step=max(0, state["step"] - 1),
                from_concept=suppress_label,
                to_concept=CONCEPTS[profile.target_concept].label,
                strength=float(delta.norm().detach().float().cpu()),
                reason=(
                    "Forward hook added target activation, subtracted suppressor activation, "
                    "and constrained the decoder readout to the approved control phrase."
                ),
            )
        )
        if isinstance(output, tuple):
            return (edited, *output[1:])
        return edited

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

    def _score_hidden(self, hidden: Any, concept_ids: list[str]) -> list[ConceptScore]:
        assert self.bank is not None and self.torch is not None
        scores: list[ConceptScore] = []
        hidden_f = hidden.detach().float()
        hidden_f = hidden_f / (hidden_f.norm() + 1e-6)
        for concept_id in concept_ids:
            concept = CONCEPTS[concept_id]
            direction = self.bank.directions[concept_id].to(hidden_f.device)
            cosine = float((hidden_f * direction).sum().cpu())
            score = max(0.0, min(1.0, (cosine + 0.25) / 0.75))
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

    def _transport_jacobian_residual(self, residual: Any) -> Any:
        assert self.jacobian_lens is not None and self.torch is not None
        original_shape = tuple(residual.shape)
        with self.torch.no_grad():
            residual_batch = residual.detach().reshape(-1, original_shape[-1]).to(
                device=self.jacobian_lens.matrix.device,
                dtype=self.jacobian_lens.matrix.dtype,
            )
            transported = residual_batch @ self.jacobian_lens.matrix.T
        if len(original_shape) == 1:
            return transported[0].float()
        return transported.reshape(*original_shape[:-1], transported.shape[-1]).float()

    def _jacobian_logits(self, transported: Any) -> Any:
        assert self.jacobian_lens is not None and self.torch is not None
        with self.torch.no_grad():
            residual_batch = transported.detach().reshape(-1, transported.shape[-1]).to(
                device=self.jacobian_lens.matrix.device,
                dtype=self.jacobian_lens.matrix.dtype,
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
