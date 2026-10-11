# J-Steer

Monitor and steer Qwen3-1.7B through hidden-state interventions using Anthropic’s [Jacobian lens](https://github.com/anthropics/jacobian-lens). FastAPI and a Tailwind UI run on Modal with a GPU-backed model.

**[Live demo](https://koogle-frick--radiant-jsteer.modal.run)**

- **Tripwire:** flag or stop generation when selected tokens appear in per-layer lens readouts.
- **Rewrite:** swap source and replacement token coordinates in the hidden state, with an unedited baseline and decoder-only control for comparison.
- Inspect token-by-layer heatmaps, intervention diagnostics, and exported run traces.

This is a research demo, not a production safety boundary. Don’t enter real secrets or private data; exported traces include the full prompt and output.

## Run

Requires Python 3.12+, `uv`, Node 22+, and configured Modal credentials (`modal setup`).

```bash
uv sync --locked --group dev
npm ci --prefix frontend
npm run build --prefix frontend
uv run modal serve -m modal_app
```

To deploy:

```bash
uv run modal deploy -m modal_app
```

The default deployment uses an L4 GPU and the matching pre-fitted Qwen3-1.7B lens from [Neuronpedia](https://huggingface.co/neuronpedia/jacobian-lens). Model and lens checkpoints must match.

## More

- [Experiments and evaluation](EXPERIMENTS.md)
- [Development and checks](CONTRIBUTING.md)
- [Related work](RELATED_WORK.md)
- [Security](SECURITY.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

Licensed under [Apache-2.0](LICENSE).
