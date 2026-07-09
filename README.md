# J-Steer

J-Steer is a demo control plane for monitoring and steering an open Qwen model through hidden-state interventions. It is designed to run on Modal by default: FastAPI serves the API and built HTML/Tailwind demo, while the model runs on a GPU-backed Modal function.

## What This Implements

This repo uses a practical J-space proxy for open weights:

- Monitor mid-layer Qwen hidden states against calibrated concept directions.
- Kill a run before completion tokens are released when a configured tripwire crosses threshold.
- Rewrite generation with a forward hook that adds a target concept direction and subtracts suppressor directions at the monitored layer.
- Show the demo as two tabs: `Tripwire` and `Rewrite`, including activation traces, readouts, intervention ledger, and baseline-vs-steered text.

Anthropic's paper describes J-space as Jacobian-lens patterns inside Claude. This project does not reproduce Anthropic's private Claude J-lens. It demonstrates the same operational control idea on Qwen using hidden-state probes and activation steering.

## Sources

- Anthropic overview: https://www.anthropic.com/research/global-workspace
- Anthropic paper: https://transformer-circuits.pub/2026/workspace/index.html
- Qwen3 Transformers quickstart: https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html
- Qwen3 0.6B model notes: https://huggingface.co/Qwen/Qwen3-0.6B
- Modal web functions: https://modal.com/docs/guide/webhooks
- Modal local file inclusion: https://modal.com/docs/guide/images

## Requirements

- Python 3.12+
- `uv`
- Node 22+
- Modal credentials already configured with `modal setup` or Modal token environment variables

## Install

```bash
uv sync --group dev
cd frontend && npm install && cd ..
```

## Serve On Modal

Build the Tailwind HTML app first, then serve the FastAPI app and Qwen engine on Modal:

```bash
cd frontend && npm run build && cd ..
uv run modal serve -m modal_app
```

Modal prints the public web URL. Open that URL for the recordable demo.

## Deploy On Modal

```bash
cd frontend && npm run build && cd ..
uv run modal deploy -m modal_app
```

Defaults:

- Model: `Qwen/Qwen3-0.6B`
- GPU: `L4`
- Hugging Face cache volume: `j-steer-hf-cache`
- Target layer: 58% through the decoder stack unless `JSTEER_LAYER` is set

## Development Checks

```bash
uv run pytest
cd frontend && npm run build && cd ..
uv run ruff check .
```

The tests inject the deterministic mock engine so CI does not need a GPU. Product serving uses the real Qwen engine through Modal by default.

## Project Layout

```text
src/jsteer/
  api.py          FastAPI app factory and static asset serving
  concepts.py     Tripwire concepts, steering profiles, demo scenarios
  hf_engine.py    Qwen hidden-state probes and forward-hook steering
  mock_engine.py  Deterministic test fixture
  schemas.py      Pydantic API contracts
frontend/
  index.html      HTML entry
  src/main.ts     Tailwind-rendered demo UI
modal_app.py      Modal ASGI deployment entrypoint
```
