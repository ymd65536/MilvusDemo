from fastapi.testclient import TestClient

from app.main import app, get_service


class FakeService:
    def health(self):
        return {"status": "ok", "milvus_uri_reachable": True, "collections": ["demo"]}

    def list_collections(self):
        return ["demo"]

    def search(self, collection, text, filter_expression, top_k, search_config):
        return {
            "results": [
                {
                    "id": "doc-1",
                    "distance": 0.91,
                    "entity": {"text": text, "category": "search"},
                }
            ],
            "latency_ms": 1.25,
        }

    def query(
        self, collection, filter_expression, output_fields, limit, offset
    ):
        return [{"id": f"doc-{offset}", "category": "search"}]


app.dependency_overrides[get_service] = lambda: FakeService()
client = TestClient(app)


def test_health_reports_milvus_status():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["collections"] == ["demo"]


def test_search_passes_top_k_filter_and_returns_scores():
    response = client.post(
        "/search",
        json={
            "collection": "demo",
            "text": "keyword and semantic search",
            "filter": 'category == "search"',
            "top_k": 3,
            "search_config": "hnsw",
        },
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["distance"] == 0.91
    assert response.json()["results"][0]["entity"]["category"] == "search"


def test_query_exposes_pagination_cursor():
    response = client.post(
        "/query",
        json={"collection": "demo", "filter": "year >= 2024", "limit": 2, "offset": 4},
    )

    assert response.status_code == 200
    assert response.json()["offset"] == 4
    assert response.json()["next_offset"] == 5


def test_collection_name_is_validated():
    response = client.post(
        "/collections",
        json={"name": "invalid-name", "dimension": 384, "enable_bm25": True},
    )

    assert response.status_code == 422
