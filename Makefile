.PHONY: install lint fmt test

install:
	python3 -m venv .venv && .venv/bin/pip install -q -U pip && .venv/bin/pip install -e ".[dev]"

lint:
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests

fmt:
	.venv/bin/ruff format src tests
	.venv/bin/ruff check --fix src tests

test:
	.venv/bin/pytest -q
