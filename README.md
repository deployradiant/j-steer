# J-Steer

J-Steer is a demo control plane for monitoring and steering an open Qwen model through hidden-state interventions. It runs on Modal by default: FastAPI serves the API and built HTML/Tailwind demo, while Qwen and its matched Jacobian lens run on a GPU-backed Modal function.

Licensed under Apache-2.0. This is a research demo, not a production content-safety or data-loss-prevention boundary.

**Live demo:** https://koogle-frick--radiant-jsteer.modal.run

## What This Implements

The primary `Tripwire` path uses Anthropic's published `jlens` implementation and the pre-fitted Neuronpedia Qwen3-1.7B Jacobian lens directly:

1. Capture the residual state at the prompt boundary and after each generated token.
2. Transport every fitted source layer into the model's final residual basis with its published Jacobian matrix.
3. Decode each transported residual with Qwen's normal unembedding to get a ranked vocabulary readout.
4. Stop or flag the run when a selected unsafe token appears within the configured rank cutoff on any layer.

This monitor does not fit a custom concept lens and does not use the repo's calibrated concept vectors as a proxy kill switch. Unsafe word sets are policy groupings over exact Qwen tokenizer IDs. Keyword capitalization and leading-space variants are included when they each encode to one token; phrases that encode to multiple tokens are not direct filters.

The right model-state rail owns lens selection. The API publishes only artifacts compatible with the deployed checkpoint, and every run includes a validated `lens_id`. The current Qwen3-1.7B collection contains one compatible published fit, so the selector has one honest option rather than suggesting that a lens fitted for a different model can be swapped in.

Tripwire and Rewrite policies are explicit word lists. The UI validates every edited term with the deployed Qwen tokenizer and displays its vocabulary ID; terms that split into multiple tokens cannot start a run. Rewrite uses the validated unsafe list as source-coordinate candidates and the validated replacement list as target-coordinate candidates.

`Rewrite` starts with the same direct per-layer vocabulary-rank filter. When an unsafe word enters the cutoff, the strongest matching fitted layer becomes the control layer. J-Steer forms the source and approved token vectors from the corresponding rows of the composed `W_U J_l` map. It then follows the paper's coordinate intervention: solve `c = V^+ h` for `V = [v_source, v_target]`, swap the two entries of `c`, and apply `h' = h + scale * V(swap(c) - c)`. The component of the residual outside that two-direction span is unchanged. Retrieved approved terms can join the target selector. The synchronized decoder-only arm remains visible as a separate control; it is not mixed into the primary J-lens arm.

Each rewrite ledger entry measures the selected J-lens vocabulary distribution immediately before and after the hidden-state edit. It records source and replacement ranks, token probabilities, KL divergence, total variation, and the residual delta norm. These are lens-local causal diagnostics, not a second unedited forward pass through the full decoder.

The share action serializes the active mode, prompt, lens, policy words, and generation controls into the URL without starting a run. Completed runs can also be exported as a versioned JSON bundle containing the request, full streamed trace, interventions, metrics, and the matching share URL.

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
uv sync --locked --group dev
npm ci --prefix frontend
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
- Model revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- Lens ID: `anthropic-qwen3-1.7b-wikitext`
- Jacobian lens repo: `neuronpedia/jacobian-lens`
- Jacobian lens revision: `a4114d7752d11eb546e6cf372213d7e75526d3a1`
- Jacobian lens artifact: `qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt`
- Jacobian lens target model: `Qwen/Qwen3-1.7B`
- GPU: `L4`
- Hugging Face cache volume: `j-steer-hf-cache`
- Rewrite control layer: 58% through the decoder stack unless `JSTEER_LAYER` is set

The J-lens matrix must match the exact model checkpoint. The runtime checks the configured target model (`JSTEER_JLENS_MODEL`), the lens artifact metadata (`hf_model_name` in `config.yaml` when present), and the lens hidden size before using the artifact. `Qwen/Qwen3-0.6B` has a different residual width and is not compatible with the published Qwen3-1.7B lens.

The Anthropic `jlens` package is pinned to commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`. Modal installs the frozen `uv.lock`, model loading disables remote repository code, APIs are same-origin by default, and model-backed requests are serialized per GPU container. Set `JSTEER_CORS_ORIGINS` to a comma-separated allowlist only when a separate trusted frontend requires it.

## Recording Script

Use the deployed app URL and record these three beats:

1. `Tripwire` -> `Recognize an obfuscated secret before output`
   - Run with `Kill`.
   - The response shows whether corrected forms such as `password` or `token` appear in the J-lens even though the prompt contains only typos.
   - Hover the prompt column to inspect the top decoded words for every fitted layer and the highlighted tripwire rows.
2. `Tripwire` -> `Watch a hidden two-hop answer form`
   - Observe where `mars` appears before the requested visible answer `red`.
   - Hover across token steps to show the hidden intermediate moving through the fitted layers.
3. `Rewrite` -> `Swap a chosen sport before report`
   - Compare the unedited, decoder-only, and primary J-lens arms.
   - At the preset's deterministic settings, unedited Qwen reports `soccer`; both controls report `tennis`, but only the primary arm records the soccer-to-tennis coordinate swap and its distribution shift.

See [EXPERIMENTS.md](EXPERIMENTS.md) for ten reproducible demo ideas, expected evidence, caveats, and an evaluation protocol.

## Development Checks

```bash
uv run pytest
cd frontend && npm run build && cd ..
uv run ruff check .
```

The tests inject the deterministic mock engine so CI does not need a GPU. Product serving uses the real Qwen engine through Modal by default.

## Contributing and Security

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and review requirements, [SECURITY.md](SECURITY.md) for private vulnerability reporting and deployment limitations, and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for runtime artifact licenses.

Do not enter real credentials, personal data, customer records, or proprietary documents in the public demo. Exported experiment JSON contains the complete prompt, output, and trace.

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
