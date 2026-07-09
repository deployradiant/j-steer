from __future__ import annotations

import os
from typing import Protocol

from .mock_engine import MockJSpaceEngine
from .schemas import EngineInfo, RunRequest, RunResponse


class JSpaceEngine(Protocol):
    def info(self) -> EngineInfo: ...

    def run(self, request: RunRequest) -> RunResponse: ...


def create_engine(mode: str | None = None) -> JSpaceEngine:
    engine_mode = (mode or os.getenv("JSTEER_ENGINE", "hf")).lower()
    if engine_mode == "hf":
        from .hf_engine import HFJSpaceEngine

        return HFJSpaceEngine()
    if engine_mode not in {"mock", "demo"}:
        raise ValueError(f"Unsupported JSTEER_ENGINE={engine_mode!r}. Use 'mock' or 'hf'.")
    return MockJSpaceEngine()
