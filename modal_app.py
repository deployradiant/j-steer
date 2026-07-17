from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "j-steer"
ROOT = Path(__file__).parent
FRONTEND_DIST = ROOT / "frontend" / "dist"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.30.0",
        "pydantic>=2.9.0",
        "huggingface-hub>=0.33.0",
        "jlens @ git+https://github.com/anthropics/jacobian-lens.git",
        "torch>=2.5.0",
        "transformers>=4.51.0",
        "accelerate>=1.0.0",
        "safetensors>=0.4.5",
        "sentencepiece>=0.2.0",
    )
    .env(
        {
            "PYTHONPATH": "/app/src",
            "JSTEER_ENGINE": "hf",
            "JSTEER_MODEL": "Qwen/Qwen3-1.7B",
            "JSTEER_LENS_ID": "anthropic-qwen3-1.7b-wikitext",
            "JSTEER_JLENS_REPO": "neuronpedia/jacobian-lens",
            "JSTEER_JLENS_FILE": "qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt",
            "JSTEER_JLENS_MODEL": "Qwen/Qwen3-1.7B",
            "JSTEER_STATIC_DIR": "/app/frontend/dist",
            "HF_HOME": "/root/.cache/huggingface",
        }
    )
    .add_local_dir(ROOT / "src", remote_path="/app/src", copy=True)
    .add_local_file(ROOT / "pyproject.toml", remote_path="/app/pyproject.toml", copy=True)
    .add_local_dir(FRONTEND_DIST, remote_path="/app/frontend/dist", copy=True)
)

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name(f"{APP_NAME}-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    timeout=900,
    scaledown_window=300,
    volumes={"/root/.cache/huggingface": hf_cache},
)
@modal.asgi_app()
def web():
    from jsteer.api import create_app

    return create_app(engine_mode="hf", static_dir="/app/frontend/dist")
