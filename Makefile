PY ?= .venv/bin/python
UV ?= uv

.PHONY: install ingest all pipeline test lint smoke notebooks clean

install:            ## create .venv (Python 3.11) and install pinned deps
	$(UV) venv --python 3.11 .venv
	$(UV) pip install --python $(PY) -r requirements.txt

ingest:             ## fetch raw NHL data for config.yaml seasons (idempotent, cached)
	$(PY) -m src.ingestion.nhl_api

pipeline:           ## everything after ingestion
	$(PY) -m src.pipeline

all: ingest pipeline test notebooks   ## full reproduction from scratch

test:
	$(PY) -m pytest tests -q

smoke:              ## tiny end-to-end run on a handful of games (no full data needed)
	$(PY) -m src.pipeline --smoke

lint:
	$(PY) -m ruff check src tests
	$(PY) -m mypy src --ignore-missing-imports

notebooks:          ## execute notebooks against the pipeline outputs
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

clean:              ## remove derived data (keeps the raw snapshot)
	rm -rf data/processed/* data/value_surfaces/* data/outputs/* reports/metrics/* reports/figures/*
