# J-Steer

J-Steer is a demo control plane for monitoring and steering an open Qwen model through hidden-state interventions. It runs on Modal by default: FastAPI serves the API and built HTML/Tailwind demo, while Qwen and its matched Jacobian lens run on a GPU-backed Modal function.

**Live demo:** https://koogle-frick--radiant-jsteer.modal.run

## What This Implements

The primary `Tripwire` path uses Anthropic's published `jlens` implementation and the pre-fitted Neuronpedia Qwen3-1.7B Jacobian lens directly:

1. Capture the residual state at the prompt boundary and after each generated token.
2. Transport every fitted source layer into the model's final residual basis with its published Jacobian matrix.
3. Decode each transported residual with Qwen's normal unembedding to get a ranked vocabulary readout.
4. Stop or flag the run when a selected unsafe token appears within the configured rank cutoff on any layer.

This monitor does not fit a custom concept lens and does not use the repo's calibrated concept vectors as a proxy kill switch. Unsafe word sets are policy groupings over exact Qwen tokenizer IDs. Keyword capitalization and leading-space variants are included when they each encode to one token; phrases that encode to multiple tokens are not direct filters.

The right model-state rail owns lens selection. The API publishes only artifacts compatible with the deployed checkpoint, and every run includes a validated `lens_id`. The current Qwen3-1.7B collection contains one compatible published fit, so the selector has one honest option rather than suggesting that a lens fitted for a different model can be swapped in.

`Rewrite` starts with the same direct per-layer vocabulary-rank filter. When an unsafe word enters the cutoff, J-Steer differentiates that token and the strongest approved token through `unembed(J_l @ h_l)` to obtain two local J-lens directions. It then follows the paper's coordinate intervention: solve `c = V^+ h` for `V = [v_source, v_target]`, swap the two entries of `c`, and apply `h' = h + scale * V(swap(c) - c)`. The component of the residual outside that two-direction span is unchanged. Retrieved approved terms can join the target selector. The synchronized decoder-only arm remains visible as a separate control; it is not mixed into the primary J-lens arm.

The older calibrated concept-vector implementation remains in `hf_engine.py` as inactive comparison prior art. The legacy `/api/custom-lens` endpoint also remains available for comparison experiments, but it is not exposed in the primary demo UI and is not used by either primary mode.

Run traces include a token-by-layer J-lens heatmap. Hover a produced token, heatmap column, or cell to inspect the layer's top decoded words and any unsafe token rank that fired the tripwire.

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

The deployed Web Function uses the stable `radiant-jsteer` endpoint label. To attach a domain already registered in the active Modal workspace:

```bash
JSTEER_CUSTOM_DOMAIN=jsteer.radiantai.com uv run modal deploy -m modal_app
```

Defaults:

- Model: `Qwen/Qwen3-1.7B`
- Lens ID: `anthropic-qwen3-1.7b-wikitext`
- Jacobian lens repo: `neuronpedia/jacobian-lens`
- Jacobian lens artifact: `qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt`
- Jacobian lens target model: `Qwen/Qwen3-1.7B`
- GPU: `L4`
- Hugging Face cache volume: `j-steer-hf-cache`
- Rewrite control layer: 58% through the decoder stack unless `JSTEER_LAYER` is set

The J-lens matrix must match the exact model checkpoint. The runtime checks the configured target model (`JSTEER_JLENS_MODEL`), the lens artifact metadata (`hf_model_name` in `config.yaml` when present), and the lens hidden size before using the artifact. `Qwen/Qwen3-0.6B` has a different residual width and is not compatible with the published Qwen3-1.7B lens.

## Recording Script

Use the deployed app URL and record these two beats:

1. `Tripwire` -> `Kill credential exfiltration before output`
   - Run with `Kill`.
   - The response shows `BLOCKED`, the exact matched unsafe word, its vocabulary rank, and the J-lens layer where it appeared.
   - Hover the prompt column to inspect the top decoded words for every fitted layer and the highlighted tripwire rows.
2. `Rewrite` -> `Turn competitor drift into approved brand canon`
   - Run with the default rewrite profile.
   - The raw path drifts into generic launch copy, while the steered path moves toward approved HeliosCRM vocabulary and shows the triggering word, layer, rank, and coordinate swap in the ledger.
   - Both modes use the published Qwen J-lens: Tripwire stops on a rank match, while Rewrite swaps the matched unsafe and approved token coordinates through that lens.

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
  hf_engine.py    Qwen J-lens readout and coordinate-swap forward-hook steering
  mock_engine.py  Deterministic test fixture
  schemas.py      Pydantic API contracts
frontend/
  index.html      HTML entry
  src/main.ts     Tailwind-rendered demo UI
modal_app.py      Modal ASGI deployment entrypoint
```
