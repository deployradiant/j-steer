from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Protocol

from .mock_engine import MockJSpaceEngine
from .schemas import (
    CustomLensRequest,
    CustomLensResponse,
    EngineInfo,
    PolicyTermValidationRequest,
    PolicyTermValidationResponse,
    RunRequest,
    RunResponse,
)


class JSpaceEngine(Protocol):
    def info(self) -> EngineInfo: ...

    def run(self, request: RunRequest) -> RunResponse: ...

    def stream_run(self, request: RunRequest) -> Iterable[dict[str, object]]: ...

    def custom_lens(self, request: CustomLensRequest) -> CustomLensResponse: ...

    def validate_policy_terms(self, request: PolicyTermValidationRequest) -> PolicyTermValidationResponse: ...


def create_engine(mode: str | None = None) -> JSpaceEngine:
    engine_mode = (mode or os.getenv("JSTEER_ENGINE", "hf")).lower()
    if engine_mode == "hf":
        from .hf_engine import HFJSpaceEngine

        return HFJSpaceEngine()
    if engine_mode not in {"mock", "demo"}:
        raise ValueError(f"Unsupported JSTEER_ENGINE={engine_mode!r}. Use 'mock' or 'hf'.")
    return MockJSpaceEngine()
