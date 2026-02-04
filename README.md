# creator-search-10m (demo)

## Start OpenSearch
docker compose up -d

## Install deps (uv)
uv sync

## Simulate data
uv run python scripts/simulate_data.py

## Create index + ingest
uv run python scripts/opensearch_create_index.py
uv run python scripts/opensearch_bulk_ingest.py

## Query
uv run creator-search search --request request.json --candidate-k 2000 --k 10

## Offline eval + latency
uv run creator-search eval --evalset evalset.jsonl --candidate-k 2000 --k 10
uv run creator-search latency --evalset evalset.jsonl --candidate-k 2000 --k 10 --n 50

## Labeling estimate
uv run creator-search labeling --requests 2000 --pairs-per-request 20 --cost-per-label 0.08

## Tests
uv run pytest -q
