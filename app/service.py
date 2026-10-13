import logging
import time
from typing import Any

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    WeightedRanker,
)

from app.config import Settings
from app.embeddings import TextEmbedder
from app.models import Document, IndexCreate

logger = logging.getLogger(__name__)

OUTPUT_FIELDS = ["id", "text", "category", "rating", "year", "metadata"]


class MilvusService:
    def __init__(self, settings: Settings) -> None:
        kwargs: dict[str, Any] = {
            "uri": settings.milvus_uri,
            "db_name": settings.milvus_db_name,
        }
        if settings.milvus_token:
            kwargs["token"] = settings.milvus_token
        self.client = MilvusClient(**kwargs)
        self.embedder = TextEmbedder(
            settings.embedding_model, settings.embedding_dimension
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "milvus_uri_reachable": True,
            "collections": self.client.list_collections(),
        }

    def create_collection(
        self, name: str, dimension: int, enable_bm25: bool
    ) -> dict[str, Any]:
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
            description="Documents for Milvus feature validation",
        )
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=4096,
            enable_analyzer=enable_bm25,
            enable_match=enable_bm25,
        )
        schema.add_field("category", DataType.VARCHAR, max_length=128)
        schema.add_field("rating", DataType.DOUBLE)
        schema.add_field("year", DataType.INT64)
        schema.add_field("metadata", DataType.JSON)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dimension)
        if enable_bm25:
            schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
            schema.add_function(
                Function(
                    name="text_bm25",
                    input_field_names=["text"],
                    output_field_names=["sparse"],
                    function_type=FunctionType.BM25,
                )
            )
        self.client.create_collection(collection_name=name, schema=schema)
        return self.client.describe_collection(collection_name=name)

    def list_collections(self) -> list[str]:
        return self.client.list_collections()

    def describe_collection(self, name: str) -> dict[str, Any]:
        return self.client.describe_collection(collection_name=name)

    def collection_action(self, name: str, action: str) -> dict[str, str]:
        if action == "load":
            self.client.load_collection(collection_name=name)
        else:
            self.client.release_collection(collection_name=name)
        return {"collection": name, "action": action}

    def drop_collection(self, name: str) -> dict[str, str]:
        self.client.drop_collection(collection_name=name)
        return {"dropped": name}

    def _rows(self, documents: list[Document]) -> list[dict[str, Any]]:
        vectors = self.embedder.encode([document.text for document in documents])
        return [
            {**document.model_dump(), "embedding": vector}
            for document, vector in zip(documents, vectors, strict=True)
        ]

    def write_documents(
        self,
        collection: str,
        documents: list[Document],
        partition_name: str | None,
        upsert: bool,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        operation = self.client.upsert if upsert else self.client.insert
        kwargs: dict[str, Any] = {
            "collection_name": collection,
            "data": self._rows(documents),
        }
        if partition_name:
            kwargs["partition_name"] = partition_name
        result = operation(**kwargs)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "%s %d documents in %.2f ms",
            "Upserted" if upsert else "Inserted",
            len(documents),
            latency_ms,
        )
        return {"result": result, "count": len(documents), "latency_ms": latency_ms}

    def delete_documents(self, collection: str, ids: list[str]) -> dict[str, Any]:
        return self.client.delete(collection_name=collection, ids=ids)

    def get_documents(self, collection: str, ids: list[str]) -> list[dict[str, Any]]:
        return self.client.get(
            collection_name=collection, ids=ids, output_fields=OUTPUT_FIELDS
        )

    def query(
        self,
        collection: str,
        filter_expression: str,
        output_fields: list[str],
        limit: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        return self.client.query(
            collection_name=collection,
            filter=filter_expression,
            output_fields=output_fields,
            limit=limit,
            offset=offset,
        )

    @staticmethod
    def _search_params(config: str) -> dict[str, Any]:
        if config == "hnsw":
            return {"metric_type": "COSINE", "params": {"ef": 64}}
        return {"metric_type": "COSINE", "params": {}}

    def search(
        self,
        collection: str,
        text: str,
        filter_expression: str,
        top_k: int,
        search_config: str,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        results = self.client.search(
            collection_name=collection,
            data=self.embedder.encode([text]),
            anns_field="embedding",
            filter=filter_expression,
            limit=top_k,
            search_params=self._search_params(search_config),
            output_fields=OUTPUT_FIELDS,
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info("Vector search returned top %d in %.2f ms", top_k, latency_ms)
        return {"results": results[0] if results else [], "latency_ms": latency_ms}

    def hybrid_search(
        self,
        collection: str,
        text: str,
        filter_expression: str,
        top_k: int,
        dense_weight: float,
        sparse_weight: float,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        common = {"expr": filter_expression} if filter_expression else {}
        dense = AnnSearchRequest(
            data=self.embedder.encode([text]),
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {}},
            limit=top_k,
            **common,
        )
        sparse = AnnSearchRequest(
            data=[text],
            anns_field="sparse",
            param={"metric_type": "BM25", "params": {}},
            limit=top_k,
            **common,
        )
        results = self.client.hybrid_search(
            collection_name=collection,
            reqs=[dense, sparse],
            ranker=WeightedRanker(dense_weight, sparse_weight),
            limit=top_k,
            output_fields=OUTPUT_FIELDS,
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info("Hybrid search returned top %d in %.2f ms", top_k, latency_ms)
        return {"results": results[0] if results else [], "latency_ms": latency_ms}

    def full_text_search(
        self, collection: str, text: str, filter_expression: str, top_k: int
    ) -> dict[str, Any]:
        started = time.perf_counter()
        results = self.client.search(
            collection_name=collection,
            data=[text],
            anns_field="sparse",
            filter=filter_expression,
            limit=top_k,
            search_params={"metric_type": "BM25", "params": {}},
            output_fields=OUTPUT_FIELDS,
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info("Full-text search returned top %d in %.2f ms", top_k, latency_ms)
        return {"results": results[0] if results else [], "latency_ms": latency_ms}

    def create_index(self, request: IndexCreate) -> dict[str, Any]:
        params = dict(request.params)
        if request.index_type == "HNSW" and not params:
            params = {"M": 16, "efConstruction": 200}
        if request.index_type == "IVF_FLAT" and not params:
            params = {"nlist": 128}
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=request.field_name,
            index_name=request.index_name,
            index_type=request.index_type,
            metric_type=request.metric_type,
            params=params,
        )
        self.client.create_index(
            collection_name=request.collection, index_params=index_params
        )
        return self.client.describe_index(
            collection_name=request.collection, index_name=request.index_name
        )

    def list_indexes(self, collection: str) -> list[str]:
        return self.client.list_indexes(collection_name=collection)

    def drop_index(self, collection: str, index_name: str) -> dict[str, str]:
        self.client.drop_index(collection_name=collection, index_name=index_name)
        return {"dropped": index_name}

    def create_partition(self, collection: str, partition_name: str) -> dict[str, str]:
        self.client.create_partition(
            collection_name=collection, partition_name=partition_name
        )
        return {"created": partition_name}

    def list_partitions(self, collection: str) -> list[str]:
        return self.client.list_partitions(collection_name=collection)

    def drop_partition(self, collection: str, partition_name: str) -> dict[str, str]:
        self.client.drop_partition(
            collection_name=collection, partition_name=partition_name
        )
        return {"dropped": partition_name}
