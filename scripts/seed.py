import json
from pathlib import Path

from app.config import get_settings
from app.models import Document, IndexCreate
from app.service import MilvusService


def main() -> None:
    settings = get_settings()
    service = MilvusService(settings)
    collection = settings.default_collection

    if collection in service.list_collections():
        service.drop_collection(collection)
    service.create_collection(collection, settings.embedding_dimension, True)
    service.create_index(
        IndexCreate(
            collection=collection,
            index_name="embedding_auto",
            index_type="AUTOINDEX",
        )
    )
    service.create_index(
        IndexCreate(
            collection=collection,
            field_name="sparse",
            index_name="sparse_bm25",
            index_type="AUTOINDEX",
            metric_type="BM25",
        )
    )
    service.create_partition(collection, "featured")

    path = Path(__file__).parents[1] / "data" / "sample_documents.json"
    documents = [Document.model_validate(item) for item in json.loads(path.read_text())]
    first_batch = service.write_documents(collection, documents[:6], None, False)
    second_batch = service.write_documents(
        collection, documents[6:], "featured", False
    )
    service.collection_action(collection, "load")
    total_latency = first_batch["latency_ms"] + second_batch["latency_ms"]
    print(
        f"Seeded {first_batch['count'] + second_batch['count']} documents into "
        f"{collection} in {total_latency:.2f} ms"
    )


if __name__ == "__main__":
    main()
