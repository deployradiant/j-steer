# Contributing

J-Steer is a research demo for inspecting and intervening on Qwen residual states. Contributions should preserve the distinction between measured model behavior, mock behavior, and interface-only visualization.

## Development setup

```bash
uv sync --locked --group dev
npm ci --prefix frontend
```

Run the required checks before opening a pull request:

```bash
uv run ruff check .
uv run pytest
npm run build --prefix frontend
```

The test suite uses the deterministic mock engine and does not require a GPU. Exercise model-backed changes on Modal with the exact pinned Qwen and lens revisions before requesting review.

## Pull requests

- Keep changes scoped and explain the behavioral claim being changed.
- Add focused tests for API contracts, token policy validation, and intervention math.
- Include screenshots for visual changes at desktop and narrow viewports.
- Do not commit model weights, lens artifacts, generated experiment exports, credentials, customer data, or real secrets.
- Do not describe calibrated concept vectors or mock data as Anthropic J-lens measurements.
- Update `THIRD_PARTY_NOTICES.md` when adding a runtime artifact or copied code.

Unless explicitly marked otherwise, contributions submitted for inclusion are licensed under Apache-2.0.
