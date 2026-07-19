from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "j-steer"
WEB_LABEL = "radiant-jsteer"
CUSTOM_DOMAIN = os.getenv("JSTEER_CUSTOM_DOMAIN")
ROOT = Path(__file__).parent
FRONTEND_DIST = ROOT / "frontend" / "dist"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_sync(uv_project_dir=".", frozen=True, uv_version="0.11.29")
    .env(
        {
            "PYTHONPATH": "/app/src",
            "JSTEER_ENGINE": "hf",
            "JSTEER_MODEL": "Qwen/Qwen3-1.7B",
            "JSTEER_MODEL_REVISION": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            "JSTEER_LENS_ID": "anthropic-qwen3-1.7b-wikitext",
            "JSTEER_JLENS_REPO": "neuronpedia/jacobian-lens",
            "JSTEER_JLENS_REVISION": "a4114d7752d11eb546e6cf372213d7e75526d3a1",
            "JSTEER_JLENS_FILE": "qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt",
            "JSTEER_JLENS_MODEL": "Qwen/Qwen3-1.7B",
            "JSTEER_STATIC_DIR": "/app/frontend/dist",
            "HF_HOME": "/root/.cache/huggingface",
        }
    )
    .add_local_dir(ROOT / "src", remote_path="/app/src", copy=True)
    .add_local_dir(FRONTEND_DIST, remote_path="/app/frontend/dist", copy=True)
)

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name(f"{APP_NAME}-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    timeout=900,
    max_containers=2,
    scaledown_window=300,
    volumes={"/root/.cache/huggingface": hf_cache},
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app(
    label=WEB_LABEL,
    custom_domains=[CUSTOM_DOMAIN] if CUSTOM_DOMAIN else None,
)
def web():
    from jsteer.api import create_app

    return create_app(engine_mode="hf", static_dir="/app/frontend/dist")
