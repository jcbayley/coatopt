.PHONY: install test lint check

UV := uv
PYTEST_ARGS ?=

install:
	$(UV) sync --extra dev

test:
	$(UV) run --extra dev pytest $(PYTEST_ARGS)

lint:
	$(UV) run --extra dev pre-commit run --all-files

check: lint test
