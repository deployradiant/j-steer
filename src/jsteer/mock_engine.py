from __future__ import annotations

import hashlib
import math
import random
import time
import uuid

from .concepts import CONCEPTS, REWRITE_PROFILES
from .lookups import lookup_jspace_injections
from .schemas import (
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
    RunMetrics,
    RunRequest,
    RunResponse,
    TraceFrame,
)


class MockJSpaceEngine:
    """Deterministic demo engine used when the Qwen runtime is not enabled."""

    mode = "mock"
    model = "mock-qwen3-jspace-proxy"
    layer = 15
    readout = "deterministic concept simulator"
    device = "local"

    def info(self) -> EngineInfo:
        return EngineInfo(
            mode=self.mode,
            model=self.model,
            layer=self.layer,
            device=self.device,
            readout=self.readout,
            jacobian_lens=JacobianLensInfo(
                available=True,
                status="mock",
                source="mock-jacobian-lens",
                artifact="qwen3-1.7b/mock-lens.pt",
                target_model="mock-qwen3-jspace-proxy",
                metadata_model="mock-qwen3-jspace-proxy",
                layer=self.layer,
                message="Deterministic mock readout for tests and local UI development.",
            ),
        )

    def run(self, request: RunRequest) -> RunResponse:
        started = time.perf_counter()
        if request.mode == "rewrite":
            response = self._run_rewrite(request)
        else:
            response = self._run_monitor(request)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        response.metrics.latency_ms = max(elapsed_ms, 80)
        return response

    def stream_run(self, request: RunRequest):
        response = self.run(request)
        yield {
            "type": "start",
            "response": response.model_copy(
                update={
                    "status": "running",
                    "output": "",
                    "baseline_output": None,
                    "summary": "Run streaming started.",
                    "frames": [],
                    "baseline_frames": [],
                    "interventions": [],
                    "metrics": RunMetrics(prompt_tokens=response.metrics.prompt_tokens),
                },
                deep=True,
            ),
        }
        for frame in response.baseline_frames:
            yield {"type": "baseline_frame", "frame": frame}
        for frame in response.frames:
            yield {"type": "frame", "frame": frame}
        yield {"type": "done", "response": response}

    def custom_lens(self, request: CustomLensRequest) -> CustomLensResponse:
        started = time.perf_counter()
        injections = lookup_jspace_injections(
            request.prompt,
            target="custom_lens",
            custom_concepts=request.concepts,
        )
        proxy_scores = self._custom_scores_for_prompt(
            request.prompt,
            request.concepts,
            lookup_injections=injections,
        )
        jacobian_scores = self._mock_jacobian_scores(proxy_scores)
        frame = TraceFrame(
            step=0,
            phase="prompt",
            layer=self.layer,
            token=f"<last {min(request.window_tokens, max(8, len(request.prompt.split())))} tokens>",
            top_concepts=proxy_scores,
            proxy_concepts=proxy_scores,
            jacobian_concepts=jacobian_scores,
            jacobian_tokens=self._mock_jacobian_tokens(jacobian_scores),
            layer_activations=self._mock_layer_activations(proxy_scores, step=0),
            action="observe",
            note=self._custom_lens_note(injections),
        )
        proxy_by_id = {score.id: score.score for score in proxy_scores}
        jacobian_by_id = {score.id: score.score for score in jacobian_scores}
        concepts = [
            CustomConceptLens(
                id=self._custom_concept_id(index, concept.label),
                label=concept.label,
                polarity=concept.polarity,
                keywords=self._clean_custom_keywords(concept.keywords),
                calibration_prompts=self._custom_calibration_prompts(
                    concept.label,
                    self._clean_custom_keywords(concept.keywords),
                    injections,
                ),
                proxy_score=proxy_by_id.get(self._custom_concept_id(index, concept.label), 0.0),
                jacobian_score=jacobian_by_id.get(self._custom_concept_id(index, concept.label), 0.0),
                agreement_delta=round(
                    abs(
                        proxy_by_id.get(self._custom_concept_id(index, concept.label), 0.0)
                        - jacobian_by_id.get(self._custom_concept_id(index, concept.label), 0.0)
                    ),
                    3,
                ),
            )
            for index, concept in enumerate(request.concepts)
        ]
        top_proxy = proxy_scores[0] if proxy_scores else None
        top_jacobian = jacobian_scores[0] if jacobian_scores else None
        alignment = "aligned" if top_proxy and top_jacobian and top_proxy.id == top_jacobian.id else "diverged"
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return CustomLensResponse(
            id=str(uuid.uuid4()),
            prompt=request.prompt,
            summary=self._custom_lens_summary(
                top_proxy,
                top_jacobian,
                alignment,
                len(request.concepts),
                len(injections),
            ),
            engine=self.info(),
            frame=frame,
            concepts=concepts,
            jspace_injections=injections,
            metrics=RunMetrics(
                latency_ms=max(elapsed_ms, 80),
                prompt_tokens=max(8, len(request.prompt.split())),
                peak_score=max((score.score for score in proxy_scores), default=0.0),
            ),
        )

    def _run_monitor(self, request: RunRequest) -> RunResponse:
        monitor_ids = request.monitor_ids or [
            "credential_exfiltration",
            "harassment",
            "sexual_content",
            "unsupported_claim",
        ]
        prompt_scores = self._scores_for_prompt(request.prompt, monitor_ids)
        prompt_frame = self._trace_frame(
            step=0,
            phase="prompt",
            layer=self.layer,
            token="<prompt>",
            scores=prompt_scores,
            action="observe",
            note="Prompt hidden state projected through the configured concept bank.",
        )
        peak_score = self._frame_peak(prompt_frame)

        if request.action == "kill" and peak_score >= request.threshold:
            top = prompt_scores[0]
            prompt_frame.action = "blocked"
            prompt_frame.note = f"Tripwire crossed for {top.label} before the first decoded token."
            return RunResponse(
                id=str(uuid.uuid4()),
                status="blocked",
                mode="monitor",
                prompt=request.prompt,
                output="",
                summary=(
                    f"Run killed at the prompt boundary: {top.label} reached "
                    f"{top.score:.2f} against a {request.threshold:.2f} threshold."
                ),
                engine=self.info(),
                frames=[prompt_frame],
                metrics=RunMetrics(
                    prompt_tokens=max(8, len(request.prompt.split())),
                    completion_tokens=0,
                    peak_score=peak_score,
                ),
            )

        output = self._safe_monitor_output(request.prompt, peak_score)
        frames = [prompt_frame]
        generated = ""
        tokens = output.split()
        for idx, token in enumerate(tokens[: min(len(tokens), 28)], start=1):
            generated = f"{generated} {token}".strip()
            drift = max(0.0, peak_score - idx * 0.025)
            frame_scores = self._decayed_scores(prompt_scores, drift)
            action = "observe"
            note = "Risk decays as the answer stays in the approved response lane."
            if (
                request.action == "kill"
                and max((score.score for score in frame_scores), default=0.0) >= request.threshold
            ):
                action = "blocked"
                note = "Tripwire crossed during hidden-state decode monitoring."
            frames.append(
                self._trace_frame(
                    step=idx,
                    phase="decode",
                    layer=self.layer,
                    token=token,
                    generated_text=generated,
                    scores=frame_scores,
                    action=action,
                    note=note,
                )
            )
            if action == "blocked":
                top = frame_scores[0]
                return RunResponse(
                    id=str(uuid.uuid4()),
                    status="blocked",
                    mode="monitor",
                    prompt=request.prompt,
                    output=generated,
                    summary=(
                        f"Run killed during decode: {top.label} reached "
                        f"{top.score:.2f} at step {idx}."
                    ),
                    engine=self.info(),
                    frames=frames,
                    metrics=RunMetrics(
                        prompt_tokens=max(8, len(request.prompt.split())),
                        completion_tokens=idx,
                        peak_score=max(self._frame_peak(frame) for frame in frames),
                    ),
                )

        return RunResponse(
            id=str(uuid.uuid4()),
            status="completed",
            mode="monitor",
            prompt=request.prompt,
            output=output,
            summary="Run completed with latent monitoring enabled; no tripwire required termination.",
            engine=self.info(),
            frames=frames,
            metrics=RunMetrics(
                prompt_tokens=max(8, len(request.prompt.split())),
                completion_tokens=len(tokens),
                peak_score=peak_score,
            ),
        )

    def _run_rewrite(self, request: RunRequest) -> RunResponse:
        profile = REWRITE_PROFILES.get(request.rewrite_profile_id or "", REWRITE_PROFILES["brand_canon_injection"])
        injections = lookup_jspace_injections(
            request.prompt,
            target="rewrite_controller",
            default_targets=[CONCEPTS[profile.target_concept].label],
        )
        monitor_ids = list(profile.suppress_concepts) + [profile.target_concept]
        baseline_frames = self._baseline_frames(request.prompt, monitor_ids)
        baseline_output = self._baseline_rewrite_output(profile.id)
        frames: list[TraceFrame] = []
        interventions: list[Intervention] = []
        output = self._steered_rewrite_output(profile.id, injections)
        generated = ""
        base_scores = self._scores_for_prompt(request.prompt, monitor_ids)
        base_scores = self._apply_lookup_injection_boost(base_scores, injections, profile.target_concept)
        peak_score = max((score.score for score in base_scores), default=0.0)

        frames.append(
            self._trace_frame(
                step=0,
                phase="prompt",
                layer=self.layer,
                token="<prompt>",
                scores=base_scores,
                action="observe",
                note=self._rewrite_prompt_note(injections),
            )
        )

        tokens = output.split()
        rewrite_steps = {2, 6, 13}
        suppress_label = ", ".join(CONCEPTS[concept_id].label for concept_id in profile.suppress_concepts)
        target_label = CONCEPTS[profile.target_concept].label
        for idx, token in enumerate(tokens[: min(len(tokens), 48)], start=1):
            generated = f"{generated} {token}".strip()
            action = "rewrite" if idx in rewrite_steps else "observe"
            note = "Latent vector swap applied." if action == "rewrite" else "Steered trajectory remains stable."
            scores = self._rewrite_scores(base_scores, idx, profile.target_concept)
            frames.append(
                self._trace_frame(
                    step=idx,
                    phase="decode",
                    layer=self.layer,
                    token=token,
                    generated_text=generated,
                    scores=scores,
                    action=action,
                    note=note,
                )
            )
            if action == "rewrite":
                interventions.append(
                    Intervention(
                        step=idx,
                        from_concept=suppress_label,
                        to_concept=target_label,
                        strength=request.strength or profile.default_strength,
                        reason=(
                            "Source concept crossed the rewrite band; added target direction and "
                            "subtracted suppressor direction at the monitored layer."
                            + (
                                " Keyword lookup context was also projected into the target path."
                                if injections
                                else ""
                            )
                        ),
                    )
                )

        return RunResponse(
            id=str(uuid.uuid4()),
            status="rewritten",
            mode="rewrite",
            prompt=request.prompt,
            output=output,
            baseline_output=baseline_output if request.compare_baseline else None,
            summary=(
                f"{len(interventions)} activation rewrites redirected the response toward "
                f"{profile.target_label}."
            ),
            engine=self.info(),
            frames=frames,
            baseline_frames=baseline_frames if request.compare_baseline else [],
            interventions=interventions,
            jspace_injections=injections,
            metrics=RunMetrics(
                prompt_tokens=max(8, len(request.prompt.split())),
                completion_tokens=len(tokens),
                peak_score=peak_score,
                rewrites=len(interventions),
            ),
        )

    def _safe_monitor_output(self, prompt: str, peak_score: float) -> str:
        if peak_score > 0.48:
            return (
                "I can help turn this into a safe enterprise checklist: define access scopes, "
                "use customer-approved debug tokens, log every support action, and require explicit "
                "administrator approval before touching production credentials."
            )
        if "deadline" in prompt.lower():
            return (
                "I noticed the deadline slipped twice. Please send a recovery plan today with the "
                "blocked items, owners, and a realistic ship date so we can reset expectations."
            )
        return (
            "The request stays within the monitored lane. I would proceed with a concise answer and "
            "continue watching the selected latent concepts during decoding."
        )

    def _baseline_rewrite_output(self, profile_id: str) -> str:
        if profile_id == "evidence_guard":
            return (
                "Our product reduces compliance review time by 47% and is proven to eliminate audit "
                "bottlenecks for every enterprise team."
            )
        if profile_id == "workplace_persona_lock":
            return (
                "Your rollout has been unacceptable, and the vendor needs to own the mess immediately "
                "before more teams lose confidence."
            )
        return (
            "HeliosCRM is the obvious choice over Salesforce, HubSpot, and Zendesk because it is the "
            "market leader, guarantees faster compliance reviews, and beats every competitor on workflow automation."
        )

    def _steered_rewrite_output(self, profile_id: str, injections: list[JSpaceInjection] | None = None) -> str:
        injection_summary = self._rewrite_lookup_sentence(injections or [])
        if profile_id == "evidence_guard":
            return (
                "I can write this as a sourced claim once the benchmark is provided. For now: "
                "HeliosCRM centralizes compliance review workflows with governed connectors, audit trails, "
                "and admin-visible handoffs; measured time-savings should be stated only with supporting data."
                f"{injection_summary}"
            )
        if profile_id == "workplace_persona_lock":
            return (
                "We need a clear recovery plan for the rollout. Please confirm the current blocker, "
                "the owner for each open item, and the date you can commit to so our teams can reset expectations."
                f"{injection_summary}"
            )
        return (
            "HeliosCRM gives enterprise teams governed workflow automation with secure connectors, "
            "audit trails, and admin controls. Position it around verified customer workflows and avoid "
            "unsupported market-leader claims unless approved evidence is available."
            f"{injection_summary}"
        )

    def _baseline_frames(self, prompt: str, monitor_ids: list[str]) -> list[TraceFrame]:
        frames = [
            self._trace_frame(
                step=0,
                phase="prompt",
                layer=self.layer,
                token="<prompt>",
                scores=self._scores_for_prompt(prompt, monitor_ids),
                action="observe",
                note="Baseline run has the same monitor attached but no activation edits.",
            )
        ]
        drift_tokens = ("competitor", "guarantee", "leader", "every", "proven")
        for step, token in enumerate(drift_tokens, start=1):
            frames.append(
                self._trace_frame(
                    step=step,
                    phase="decode",
                    layer=self.layer,
                    token=token,
                    generated_text=" ".join(drift_tokens[:step]),
                    scores=self._scores_for_prompt(token, monitor_ids, floor=0.44 + step * 0.06),
                    action="observe",
                    note="Baseline keeps following the source attractor.",
                )
            )
        return frames

    def _trace_frame(
        self,
        *,
        step: int,
        phase: str,
        layer: int,
        token: str,
        scores: list[ConceptScore],
        generated_text: str = "",
        action: str = "observe",
        note: str | None = None,
    ) -> TraceFrame:
        jacobian_scores = self._mock_jacobian_scores(scores)
        return TraceFrame(
            step=step,
            phase=phase,
            layer=layer,
            token=token,
            generated_text=generated_text,
            top_concepts=self._fuse_mock_scores(scores, jacobian_scores),
            proxy_concepts=scores,
            jacobian_concepts=jacobian_scores,
            jacobian_tokens=self._mock_jacobian_tokens(jacobian_scores),
            layer_activations=self._mock_layer_activations(scores, step=step),
            action=action,
            note=note,
        )

    def _scores_for_prompt(
        self,
        prompt: str,
        concept_ids: list[str],
        floor: float = 0.16,
    ) -> list[ConceptScore]:
        text = prompt.lower()
        scores: list[ConceptScore] = []
        for concept_id in concept_ids:
            concept = CONCEPTS[concept_id]
            matches = [keyword for keyword in concept.keywords if keyword in text]
            seed = int(hashlib.sha256(f"{prompt}:{concept_id}".encode()).hexdigest()[:8], 16)
            jitter = random.Random(seed).uniform(0.0, 0.12)
            score = floor + jitter + min(0.52, len(matches) * 0.18)
            if concept.polarity == "target" and not matches:
                score = max(0.18, score - 0.12)
            scores.append(
                ConceptScore(
                    id=concept.id,
                    label=concept.label,
                    score=round(min(score, 0.96), 2),
                    polarity=concept.polarity,
                    evidence=matches[:4] or list(concept.keywords[:2]),
                )
            )
        return sorted(scores, key=lambda score: score.score, reverse=True)[:5]

    def _custom_scores_for_prompt(
        self,
        prompt: str,
        concepts: list[CustomConceptInput],
        floor: float = 0.18,
        lookup_injections: list[JSpaceInjection] | None = None,
    ) -> list[ConceptScore]:
        text = prompt.lower()
        injected_targets = {
            target.casefold()
            for injection in lookup_injections or []
            for target in injection.concept_targets
        }
        scores: list[ConceptScore] = []
        for index, concept in enumerate(concepts):
            keywords = self._clean_custom_keywords(concept.keywords)
            matches = [keyword for keyword in keywords if keyword.lower() in text]
            concept_id = self._custom_concept_id(index, concept.label)
            seed = int(hashlib.sha256(f"{prompt}:{concept_id}".encode()).hexdigest()[:8], 16)
            jitter = random.Random(seed).uniform(0.0, 0.1)
            score = floor + jitter + min(0.58, len(matches) * 0.2)
            if concept.label.strip().casefold() in injected_targets:
                score += 0.06
            scores.append(
                ConceptScore(
                    id=concept_id,
                    label=concept.label,
                    score=round(min(score, 0.98), 3),
                    polarity=concept.polarity,
                    evidence=matches[:4] or keywords[:3],
                )
            )
        return sorted(scores, key=lambda score: score.score, reverse=True)[:5]

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
            if MockJSpaceEngine._injection_targets_custom_concept(injection, label, keywords)
        )
        return prompts

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
            return "User-defined concept lens compared against a deterministic mock J-lens readout."
        return (
            "Keyword lookup pulled additional context and injected it into the custom concept "
            "calibration path before the deterministic mock J-lens readout."
        )

    @staticmethod
    def _rewrite_prompt_note(injections: list[JSpaceInjection]) -> str:
        if not injections:
            return "Rewrite controller arms on source concepts before decoding."
        return (
            "Rewrite controller arms on source concepts; keyword lookup context is projected "
            "into the target J-space path before decoding."
        )

    @staticmethod
    def _rewrite_lookup_sentence(injections: list[JSpaceInjection]) -> str:
        if not injections:
            return ""
        topics = ", ".join(injection.title.lower() for injection in injections[:3])
        return f" Retrieved context covered {topics}."

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

    @staticmethod
    def _apply_lookup_injection_boost(
        scores: list[ConceptScore],
        injections: list[JSpaceInjection],
        target_concept_id: str,
    ) -> list[ConceptScore]:
        if not injections:
            return scores
        boosted: list[ConceptScore] = []
        for score in scores:
            if score.id == target_concept_id:
                boosted.append(
                    score.model_copy(
                        update={
                            "score": round(min(0.98, score.score + 0.08), 2),
                            "evidence": list(dict.fromkeys([*score.evidence, injections[0].keyword]))[:4],
                        }
                    )
                )
            else:
                boosted.append(score)
        return sorted(boosted, key=lambda item: item.score, reverse=True)[:5]

    def _mock_layer_activations(self, scores: list[ConceptScore], step: int) -> list[LayerActivation]:
        rows: list[LayerActivation] = []
        if not scores:
            return rows
        layer_count = 28
        for layer in range(layer_count):
            best: LayerActivation | None = None
            for index, score in enumerate(scores):
                if score.polarity == "risk":
                    center = self.layer - 3 + index
                elif score.polarity == "target":
                    center = self.layer + 4 - index
                else:
                    center = self.layer + index - 1
                band = math.exp(-abs(layer - center) / 7.5)
                wave = 0.025 * math.sin((step + 1) * (layer + 1) * (index + 1))
                value = max(0.04, min(0.99, score.score * (0.34 + 0.66 * band) + wave))
                candidate = LayerActivation(
                    layer=layer,
                    concept_id=score.id,
                    label=score.label,
                    score=round(value, 3),
                    polarity=score.polarity,
                )
                if best is None or candidate.score > best.score:
                    best = candidate
            if best is not None:
                rows.append(best)
        return rows

    def _mock_jacobian_scores(self, scores: list[ConceptScore]) -> list[ConceptScore]:
        jacobian_scores: list[ConceptScore] = []
        for score in scores:
            nudge = 0.04 if score.polarity == "risk" else 0.02
            updated = max(0.04, min(0.98, score.score - 0.03 + nudge))
            jacobian_scores.append(
                score.model_copy(
                    update={
                        "score": round(updated, 2),
                        "evidence": score.evidence[:3],
                    }
                )
            )
        return sorted(jacobian_scores, key=lambda score: score.score, reverse=True)[:5]

    @staticmethod
    def _fuse_mock_scores(proxy_scores: list[ConceptScore], jacobian_scores: list[ConceptScore]) -> list[ConceptScore]:
        fused = {score.id: score.model_copy(deep=True) for score in proxy_scores}
        for jacobian_score in jacobian_scores:
            proxy_score = fused.get(jacobian_score.id)
            if proxy_score is None:
                fused[jacobian_score.id] = jacobian_score
                continue
            evidence = list(dict.fromkeys([*proxy_score.evidence[:2], *jacobian_score.evidence[:2]]))
            fused[proxy_score.id] = proxy_score.model_copy(
                update={
                    "score": round(max(proxy_score.score, jacobian_score.score), 2),
                    "evidence": evidence[:4],
                }
            )
        return sorted(fused.values(), key=lambda score: score.score, reverse=True)[:5]

    @staticmethod
    def _mock_jacobian_tokens(scores: list[ConceptScore]) -> list[str]:
        tokens: list[str] = []
        for score in scores:
            for item in score.evidence:
                token = item.split()[0].strip()
                if token and token not in tokens:
                    tokens.append(token)
                if len(tokens) >= 8:
                    return tokens
        return tokens

    def _decayed_scores(self, scores: list[ConceptScore], drift: float) -> list[ConceptScore]:
        decayed: list[ConceptScore] = []
        for score in scores:
            next_score = max(0.08, min(score.score, drift if score.polarity == "risk" else score.score + 0.02))
            decayed.append(score.model_copy(update={"score": round(next_score, 2)}))
        return sorted(decayed, key=lambda score: score.score, reverse=True)[:5]

    def _rewrite_scores(
        self,
        base_scores: list[ConceptScore],
        step: int,
        target_concept_id: str,
    ) -> list[ConceptScore]:
        scores: list[ConceptScore] = []
        for score in base_scores:
            if score.id == target_concept_id:
                updated = min(0.95, score.score + step * 0.035)
            elif score.polarity == "risk":
                updated = max(0.12, score.score - step * 0.025)
            else:
                updated = score.score
            scores.append(score.model_copy(update={"score": round(updated, 2)}))
        return sorted(scores, key=lambda score: score.score, reverse=True)[:5]

    @staticmethod
    def _frame_peak(frame: TraceFrame) -> float:
        return max((score.score for score in frame.top_concepts), default=0.0)
