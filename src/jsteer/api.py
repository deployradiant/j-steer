from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .concepts import SCENARIOS, public_concepts, public_rewrite_profiles
from .engine import create_engine
from .lenses import DEFAULT_LENS_ID, public_lenses
from .schemas import (
    CustomLensRequest,
    CustomLensResponse,
    PolicyTermValidationRequest,
    PolicyTermValidationResponse,
    ProfilesResponse,
    RunMetrics,
    RunRequest,
    RunResponse,
)


def create_app(engine_mode: str | None = None, static_dir: str | Path | None = None) -> FastAPI:
    engine = create_engine(engine_mode)
    app = FastAPI(
        title="J-Steer",
        version="0.1.0",
        description="Monitor and steer Qwen hidden-state activations for enterprise control demos.",
    )
    app.state.engine = engine

    cors_origins = [origin.strip() for origin in os.getenv("JSTEER_CORS_ORIGINS", "").split(",") if origin.strip()]
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'"
        )
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {"ok": True, "engine": engine.info().model_dump()}

    @app.get("/api/profiles", response_model=ProfilesResponse)
    def profiles() -> ProfilesResponse:
        return ProfilesResponse(
            concepts=public_concepts(),
            rewrite_profiles=public_rewrite_profiles(),
            scenarios=list(SCENARIOS),
            lenses=public_lenses(),
            default_lens_id=DEFAULT_LENS_ID,
        )

    @app.post("/api/run", response_model=RunResponse)
    async def run(request: RunRequest) -> RunResponse:
        try:
            result = engine.run(request)
            if inspect.isawaitable(result):
                result = await result
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/policy-terms/validate", response_model=PolicyTermValidationResponse)
    async def validate_policy_terms(request: PolicyTermValidationRequest) -> PolicyTermValidationResponse:
        try:
            result = engine.validate_policy_terms(request)
            if inspect.isawaitable(result):
                result = await result
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/run/stream")
    async def run_stream(api_request: Request, request: RunRequest) -> StreamingResponse:
        async def events() -> AsyncIterator[bytes]:
            try:
                stream_run = getattr(engine, "stream_run", None)
                if stream_run is None:
                    result = engine.run(request)
                    if inspect.isawaitable(result):
                        result = await result
                    stream: Iterable[dict[str, Any]] = _events_from_response(result)
                else:
                    stream = stream_run(request)

                if inspect.isasyncgen(stream):
                    async for event in stream:
                        if await api_request.is_disconnected():
                            break
                        yield _encode_stream_event(event)
                else:
                    for event in stream:
                        if await api_request.is_disconnected():
                            break
                        yield _encode_stream_event(event)
                        await asyncio.sleep(0)
            except ValueError as exc:
                yield _encode_stream_event({"type": "error", "message": str(exc), "status_code": 400})
            except RuntimeError as exc:
                yield _encode_stream_event({"type": "error", "message": str(exc), "status_code": 503})

        return StreamingResponse(events(), media_type="application/x-ndjson")

    @app.post("/api/custom-lens", response_model=CustomLensResponse)
    async def custom_lens(request: CustomLensRequest) -> CustomLensResponse:
        try:
            result = engine.custom_lens(request)
            if inspect.isawaitable(result):
                result = await result
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    mount_static(app, static_dir)
    return app


def mount_static(app: FastAPI, static_dir: str | Path | None = None) -> None:
    raw_dir = static_dir or os.getenv("JSTEER_STATIC_DIR")
    if raw_dir is None:
        candidate = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    else:
        candidate = Path(raw_dir)

    if not candidate.exists():
        return

    assets_dir = candidate / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    index_file = candidate / "index.html"

    @app.get("/", include_in_schema=False)
    def index():
        if index_file.exists():
            return FileResponse(index_file)
        return HTMLResponse("<h1>J-Steer API is running</h1>")

    @app.get("/{path:path}", include_in_schema=False)
    def spa_fallback(path: str):
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        static_file = candidate / path
        if static_file.is_file() and candidate in static_file.resolve().parents:
            return FileResponse(static_file)
        return FileResponse(index_file)


app = create_app()


def _events_from_response(response: RunResponse) -> Iterable[dict[str, Any]]:
    yield {
        "type": "start",
        "response": response.model_copy(
            update={
                "status": "running",
                "output": "",
                "baseline_output": None,
                "decoder_output": None,
                "summary": "Run streaming started.",
                "frames": [],
                "baseline_frames": [],
                "decoder_frames": [],
                "interventions": [],
                "metrics": RunMetrics(prompt_tokens=response.metrics.prompt_tokens),
            },
            deep=True,
        ),
    }
    for frame in response.baseline_frames:
        yield {"type": "baseline_frame", "frame": frame}
    for frame in response.decoder_frames:
        yield {"type": "decoder_frame", "frame": frame}
    for frame in response.frames:
        yield {"type": "frame", "frame": frame}
    yield {"type": "done", "response": response}


def _encode_stream_event(event: dict[str, Any]) -> bytes:
    return f"{json.dumps(jsonable_encoder(event), separators=(',', ':'))}\n".encode()
