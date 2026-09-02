from unittest.mock import MagicMock

from pymilvus import DataType, FunctionType, MilvusClient

from app.models import IndexCreate
from app.service import MilvusService


def service_with_fake_client() -> MilvusService:
    service = MilvusService.__new__(MilvusService)
    service.client = MagicMock()
    service.client.create_schema.side_effect = MilvusClient.create_schema
    return service


def test_create_collection_builds_dense_sparse_and_metadata_schema():
    service = service_with_fake_client()
    service.client.describe_collection.return_value = {"collection_name": "articles"}

    result = service.create_collection("articles", 384, enable_bm25=True)

    schema = service.client.create_collection.call_args.kwargs["schema"]
    fields = {field.name: field for field in schema.fields}
    assert fields["id"].is_primary is True
    assert fields["metadata"].dtype == DataType.JSON
    assert fields["embedding"].params["dim"] == 384
    assert fields["sparse"].dtype == DataType.SPARSE_FLOAT_VECTOR
    assert schema.functions[0].type == FunctionType.BM25
    assert result == {"collection_name": "articles"}


def test_create_hnsw_index_supplies_useful_defaults():
    service = service_with_fake_client()
    index_params = MagicMock()
    service.client.prepare_index_params.return_value = index_params
    service.client.describe_index.return_value = {"index_name": "embedding_hnsw"}

    result = service.create_index(
        IndexCreate(
            collection="articles",
            index_name="embedding_hnsw",
            index_type="HNSW",
        )
    )

    index_params.add_index.assert_called_once_with(
        field_name="embedding",
        index_name="embedding_hnsw",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    assert result == {"index_name": "embedding_hnsw"}
