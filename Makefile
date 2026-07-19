.PHONY: install build-ui serve deploy test lint

install:
	uv sync --locked --group dev
	npm ci --prefix frontend

build-ui:
	cd frontend && npm run build

serve: build-ui
	uv run modal serve -m modal_app

deploy: build-ui
	uv run modal deploy -m modal_app

test:
	uv run pytest
	cd frontend && npm run build

lint:
	uv run ruff check .
