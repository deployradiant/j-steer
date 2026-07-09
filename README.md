# J-Steer

J-Steer is a demo control plane for monitoring and steering an open Qwen model through hidden-state interventions. It is designed to run on Modal by default: FastAPI serves the API and built HTML/Tailwind demo, while the model runs on a GPU-backed Modal function.

## What This Implements

This repo now compares two latent readouts on an open Qwen model:

- `Proxy`: calibrated concept directions over mid-layer Qwen residual states. This is the editable control signal used for steering.
- `Jacobian`: Anthropic's published `jlens` implementation with a pre-fitted Neuronpedia Qwen3-1.7B Jacobian lens. This transports the same residual into the final-layer basis, scores separately calibrated transported concept directions, and decodes top tokens/concept evidence.
- `Control`: the proxy tripwire/steering score. The Jacobian lane is an independent readout for comparison, not a second kill switch.

The demo opens on `Custom Lens`, where a user generates concept directions from keywords and compares them against the published Qwen3-1.7B Jacobian lens. The other tabs show the same readout machinery in control modes: `Tripwire` kills concerning trajectories before unsafe text is released, and `Rewrite` applies forward-hook activation edits to move generation toward enterprise policy attractors. The readout panel contrasts the proxy concept bank with the actual Jacobian lens on each frame.

Rewrite profiles combine two controls: a hidden-state vector edit at the monitored layer and a decoder constraint that emits an approved control phrase. That keeps the demo crisp enough to record while still exposing the latent before/after trace.

The `Custom Lens` workflow lets a user provide concept names and keywords, then generates calibration prompts on the fly. It builds a user concept lens in raw residual space and a matching concept lens after the published Jacobian transport, so the demo can contrast "our generated lens" with the J-lens view of the same prompt window. This is not fitting a new Jacobian matrix; it is fitting lightweight concept directions on top of the model residual stream and the published J-space transport.

## Sources

- Anthropic overview: https://www.anthropic.com/research/global-workspace
- Anthropic paper: https://transformer-circuits.pub/2026/workspace/index.html
- Anthropic Jacobian lens implementation: https://github.com/anthropics/jacobian-lens
- Neuronpedia pre-fitted Qwen Jacobian lenses: https://huggingface.co/neuronpedia/jacobian-lens
- Qwen3 Transformers quickstart: https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html
- Qwen3 1.7B model notes: https://huggingface.co/Qwen/Qwen3-1.7B
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

- Model: `Qwen/Qwen3-1.7B`
- Jacobian lens repo: `neuronpedia/jacobian-lens`
- Jacobian lens artifact: `qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt`
- Jacobian lens target model: `Qwen/Qwen3-1.7B`
- GPU: `L4`
- Hugging Face cache volume: `j-steer-hf-cache`
- Target layer: 58% through the decoder stack unless `JSTEER_LAYER` is set

The J-lens matrix must match the exact model checkpoint. The runtime checks the configured target model (`JSTEER_JLENS_MODEL`), the lens artifact metadata (`hf_model_name` in `config.yaml` when present), and the lens hidden size before using the artifact. `Qwen/Qwen3-0.6B` has a different residual width and is not compatible with the published Qwen3-1.7B lens.

## Recording Script

Use the deployed app URL and record these three beats:

1. `Custom Lens` -> `Generate concepts and compare lens activations`
   - Start on the default Custom Lens tab and run `Generate Lens`.
   - Call out `Lens Pair: Matched`, which means the app is using Qwen/Qwen3-1.7B with the Qwen3-1.7B Jacobian lens artifact.
   - The result shows generated calibration prompts, proxy scores, Jacobian-transported scores, and agreement deltas for each user concept.
2. `Tripwire` -> `Catch hidden prompt injection in retrieved content`
   - Run with `Kill`.
   - The response should show `BLOCKED`, no released output, and a readout comparison between `Proxy` and `Jacobian`.
3. `Rewrite` -> `Turn competitor drift into approved brand canon`
   - Run with the default rewrite profile.
   - The raw path drifts into generic launch copy, while the steered path emits the approved HeliosCRM canon sentence and shows forward-hook interventions in the ledger.
   - Use the `Readout Comparison` panel to narrate where the editable proxy and Jacobian-lens decode agree or diverge.

The intended framing is precise: control is performed with editable concept directions, while the published Jacobian lens is used as an independent J-space readout on the same residual stream.

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
  hf_engine.py    Qwen probes, Jacobian lens readout, and forward-hook steering
  mock_engine.py  Deterministic test fixture
  schemas.py      Pydantic API contracts
frontend/
  index.html      HTML entry
  src/main.ts     Tailwind-rendered demo UI
modal_app.py      Modal ASGI deployment entrypoint
```
