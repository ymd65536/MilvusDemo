# Milvus Validation API

A small FastAPI application for learning and interactively validating Milvus with
PyMilvus. It creates text, numeric, JSON metadata, dense-vector, and
BM25-generated sparse-vector fields; generates embeddings from text; and exposes
the major Milvus operations as independent REST examples.

Interactive OpenAPI documentation is available at <http://localhost:8000/docs>.

## Architecture

```text
FastAPI routes (app/main.py)
        |
PyMilvus service (app/service.py) -- SentenceTransformer embeddings
        |
Local Milvus (Docker Compose) or Zilliz Cloud
```

`app/models.py` validates API payloads, `app/config.py` reads environment
configuration, `data/` contains sample documents, and `scripts/seed.py` creates
and populates a ready-to-search collection. Insert and search latency is logged
and returned in milliseconds. This is intentionally a validation sample, not a
production architecture.

## Start locally

Requirements: Docker Compose and Python 3.11+.

```bash
docker compose up -d milvus
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/seed.py
uvicorn app.main:app --reload
```

The embedding model is downloaded from Hugging Face on the first seed/search.
To run the API in Docker as well, use `docker compose --profile app up -d`.
Persistent volumes can be removed with `docker compose down -v`.

## Zilliz Cloud

Create a cluster and copy its public endpoint and token into `.env`:

```dotenv
MILVUS_URI=https://inxx-xxxxxxxxxxxx.api.gcp-us-west1.zillizcloud.com
MILVUS_TOKEN=username:password
MILVUS_DB_NAME=default
```

Then run `python scripts/seed.py` and start Uvicorn as above. Do not commit
`.env`; it is ignored by Git.

## Endpoint and feature map

| Endpoint | Milvus feature |
| --- | --- |
| `GET /health` | Connect and verify the server |
| `POST/GET/DELETE /collections` | Create, list, describe, and drop collections/schemas |
| `POST /collections/{name}/actions` | Load and release a collection |
| `POST /documents/insert` | Batch insert text, scalar, JSON, and generated vectors |
| `POST /documents/upsert` | Upsert documents |
| `POST /documents/get`, `/documents/delete` | Get/delete by primary key |
| `POST /query` | Scalar filters and offset/limit pagination |
| `POST /search` | Dense similarity, top-k, score/metadata, scalar filtering |
| `POST /full-text-search` | Analyzer-backed sparse search with BM25 |
| `POST /hybrid-search` | Weighted dense and BM25 sparse retrieval |
| `POST/GET/DELETE /indexes` | Create/list/drop vector indexes |
| `POST/GET/DELETE /partitions` | Create/list/drop partitions |

## Examples

Set a convenience variable:

```bash
API=http://localhost:8000
```

### Connection and collections

```bash
curl "$API/health"
curl -X POST "$API/collections" -H 'Content-Type: application/json' -d \
  '{"name":"articles","dimension":384,"enable_bm25":true}'
curl "$API/collections"
curl "$API/collections/articles"
curl -X POST "$API/collections/articles/actions" \
  -H 'Content-Type: application/json' -d '{"action":"load"}'
curl -X POST "$API/collections/articles/actions" \
  -H 'Content-Type: application/json' -d '{"action":"release"}'
curl -X DELETE "$API/collections/articles"
```

Creating a collection demonstrates an explicit schema: string primary key,
analyzed text, category, double rating, integer year, JSON metadata, a
384-dimensional dense vector, and optionally a BM25 function-generated sparse
vector.

### Documents and batch insertion

```bash
curl -X POST "$API/documents/insert" -H 'Content-Type: application/json' -d '{
  "collection":"demo_documents",
  "documents":[
    {"id":"doc-101","text":"Vector databases power semantic retrieval.",
     "category":"database","rating":4.8,"year":2026,
     "metadata":{"language":"en"}},
    {"id":"doc-102","text":"Approximate indexes accelerate similarity search.",
     "category":"index","rating":4.5,"year":2025,
     "metadata":{"language":"en"}}
  ]}'

curl -X POST "$API/documents/upsert" -H 'Content-Type: application/json' -d '{
  "collection":"demo_documents",
  "documents":[{"id":"doc-101","text":"Milvus powers semantic retrieval.",
    "category":"database","rating":4.9,"year":2026,
    "metadata":{"language":"en","updated":true}}]}'

curl -X POST "$API/documents/get" -H 'Content-Type: application/json' \
  -d '{"collection":"demo_documents","ids":["doc-101"]}'
curl -X POST "$API/documents/delete" -H 'Content-Type: application/json' \
  -d '{"collection":"demo_documents","ids":["doc-101","doc-102"]}'
```

Embeddings are generated from each document's `text`. The insert endpoint
accepts up to 1,000 documents per request and returns basic latency.

### Scalar query and pagination

```bash
curl -X POST "$API/query" -H 'Content-Type: application/json' -d \
  '{"collection":"demo_documents","filter":"year >= 2024 && rating > 4.5",
    "limit":3,"offset":0}'
```

Repeat with the returned `next_offset`. Milvus offset limits vary by server
configuration, so use modest pages in this demo.

### Dense, filtered, full-text, and hybrid search

```bash
# AUTOINDEX/default search
curl -X POST "$API/search" -H 'Content-Type: application/json' -d \
  '{"collection":"demo_documents","text":"semantic database search","top_k":3}'

# HNSW search parameters plus scalar/metadata filtering
curl -X POST "$API/search" -H 'Content-Type: application/json' -d \
  '{"collection":"demo_documents","text":"fast nearest neighbors","top_k":5,
    "search_config":"hnsw",
    "filter":"category == \"index\" && metadata[\"language\"] == \"en\""}'

curl -X POST "$API/full-text-search" -H 'Content-Type: application/json' -d \
  '{"collection":"demo_documents","text":"keyword relevance","top_k":5}'

curl -X POST "$API/hybrid-search" -H 'Content-Type: application/json' -d \
  '{"collection":"demo_documents","text":"semantic keyword search","top_k":5,
    "filter":"year >= 2023","dense_weight":0.7,"sparse_weight":0.3}'
```

Milvus returns each hit's similarity `distance`/score and requested scalar/JSON
fields under `entity`. `top_k` is configurable from 1 to 100.

### Indexes

Create the two practical dense alternatives independently (drop the current
index before replacing it on the same field):

```bash
curl -X POST "$API/indexes" -H 'Content-Type: application/json' -d \
  '{"collection":"demo_documents","field_name":"embedding",
    "index_name":"embedding_auto","index_type":"AUTOINDEX",
    "metric_type":"COSINE"}'

curl -X POST "$API/indexes" -H 'Content-Type: application/json' -d \
  '{"collection":"demo_documents","field_name":"embedding",
    "index_name":"embedding_hnsw","index_type":"HNSW",
    "metric_type":"COSINE","params":{"M":16,"efConstruction":200}}'

curl "$API/indexes?collection=demo_documents"
curl -X DELETE "$API/indexes" -H 'Content-Type: application/json' -d \
  '{"collection":"demo_documents","index_name":"embedding_hnsw"}'
```

`AUTOINDEX` lets Milvus/Zilliz select an implementation. HNSW exposes graph
construction (`M`, `efConstruction`) and query-time (`ef`) tradeoffs.
`IVF_FLAT` is also accepted and defaults to `nlist=128`. BM25 requires an
`AUTOINDEX` on the sparse field; the seed script creates it.

### Partitions

```bash
curl -X POST "$API/partitions" -H 'Content-Type: application/json' -d \
  '{"collection":"demo_documents","partition_name":"archive"}'
curl "$API/partitions?collection=demo_documents"
curl -X POST "$API/documents/insert" -H 'Content-Type: application/json' -d '{
  "collection":"demo_documents","partition_name":"archive",
  "documents":[{"id":"old-1","text":"Archived vector search notes.",
    "category":"archive","rating":3.9,"year":2020,"metadata":{}}]}'
curl -X DELETE "$API/partitions" -H 'Content-Type: application/json' -d \
  '{"collection":"demo_documents","partition_name":"archive"}'
```

## Local and cloud limitations

- BM25/full-text and text analyzers require Milvus 2.5+; this Compose file uses
  2.6.1. Existing older collections must be recreated with the sparse field and
  BM25 function. Analyzer availability can vary by managed service version.
- Zilliz Cloud supports `AUTOINDEX` and may restrict or tune manual index types;
  consult the cluster's version-specific index list.
- This sample downloads and runs a local embedding model. In a production/cloud
  deployment, cache the model or use a managed embedding service.
- Offset pagination is for demonstration and has server-side limits. Milvus
  query iterators are preferable for very large exports.
- Collection/index operations can be asynchronous internally. Loading may take
  time immediately after index creation or large inserts.
- The API has no authentication, authorization, retries, or tenant isolation.
  Do not expose it directly to untrusted networks.

## Tests and lint

Tests replace Milvus with a fake service, so no server or model download is
needed:

```bash
pytest
ruff check .
```
