from __future__ import annotations

import inspect
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .concepts import SCENARIOS, public_concepts, public_rewrite_profiles
from .engine import create_engine
from .schemas import ProfilesResponse, RunRequest, RunResponse


def create_app(engine_mode: str | None = None, static_dir: str | Path | None = None) -> FastAPI:
    engine = create_engine(engine_mode)
    app = FastAPI(
        title="J-Steer",
        version="0.1.0",
        description="Monitor and steer Qwen hidden-state activations for enterprise control demos.",
    )
    app.state.engine = engine

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {"ok": True, "engine": engine.info().model_dump()}

    @app.get("/api/profiles", response_model=ProfilesResponse)
    def profiles() -> ProfilesResponse:
        return ProfilesResponse(
            concepts=public_concepts(),
            rewrite_profiles=public_rewrite_profiles(),
            scenarios=list(SCENARIOS),
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
        return FileResponse(index_file)


app = create_app()
