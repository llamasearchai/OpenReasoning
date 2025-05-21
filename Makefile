.PHONY: install test lint run-api docs clean

install:
	pip install -e .

test:
	pytest tests/

lint:
	black openreasoning tests
	isort openreasoning tests

run-api:
	python -m openreasoning.cli server

run-cli:
	python -m openreasoning.cli

run-chat:
	python -m openreasoning.cli chat

docs:
	jupyter-nbconvert --to markdown notebooks/*.ipynb --output-dir docs/examples/

m3-test:
	python -c "from openreasoning.utils.m3_optimizer import m3_optimizer; print(m3_optimizer.get_optimization_status())"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf .egg-info
	find . -name pycache -type d -exec rm -rf {} +
	find . -name ".pyc" -delete 