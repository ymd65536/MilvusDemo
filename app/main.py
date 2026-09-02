import logging
from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query
from pymilvus.exceptions import MilvusException

from app.config import get_settings
from app.models import (
    CollectionAction,
    CollectionCreate,
    DeleteRequest,
    DocumentsWrite,
    FullTextSearchRequest,
    GetRequest,
    HybridSearchRequest,
    IndexCreate,
    IndexDrop,
    PartitionCreate,
    PartitionDrop,
    QueryRequest,
    SearchRequest,
)
from app.service import MilvusService

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

app = FastAPI(
    title="Milvus Validation API",
    description="Small REST API for interactively validating major Milvus features.",
    version="1.0.0",
)


@lru_cache
def get_service() -> MilvusService:
    return MilvusService(get_settings())


Service = Annotated[MilvusService, Depends(get_service)]


@app.exception_handler(MilvusException)
async def milvus_exception_handler(_request, exc: MilvusException):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/health", tags=["health"])
def health(service: Service) -> dict[str, Any]:
    try:
        return service.health()
    except Exception as exc:
        detail = f"Milvus unavailable: {exc}"
        raise HTTPException(status_code=503, detail=detail) from exc


@app.post("/collections", tags=["collections"], status_code=201)
def create_collection(request: CollectionCreate, service: Service):
    return service.create_collection(
        request.name, request.dimension, request.enable_bm25
    )


@app.get("/collections", tags=["collections"])
def list_collections(service: Service):
    return {"collections": service.list_collections()}


@app.get("/collections/{name}", tags=["collections"])
def describe_collection(name: str, service: Service):
    return service.describe_collection(name)


@app.post("/collections/{name}/actions", tags=["collections"])
def collection_action(name: str, request: CollectionAction, service: Service):
    return service.collection_action(name, request.action)


@app.delete("/collections/{name}", tags=["collections"])
def drop_collection(name: str, service: Service):
    return service.drop_collection(name)


@app.post("/documents/insert", tags=["documents"])
def insert_documents(request: DocumentsWrite, service: Service):
    return service.write_documents(
        request.collection, request.documents, request.partition_name, False
    )


@app.post("/documents/upsert", tags=["documents"])
def upsert_documents(request: DocumentsWrite, service: Service):
    return service.write_documents(
        request.collection, request.documents, request.partition_name, True
    )


@app.post("/documents/get", tags=["documents"])
def get_documents(request: GetRequest, service: Service):
    return {"documents": service.get_documents(request.collection, request.ids)}


@app.post("/documents/delete", tags=["documents"])
def delete_documents(request: DeleteRequest, service: Service):
    return service.delete_documents(request.collection, request.ids)


@app.post("/query", tags=["query"])
def query_documents(request: QueryRequest, service: Service):
    documents = service.query(
        request.collection,
        request.filter,
        request.output_fields,
        request.limit,
        request.offset,
    )
    return {
        "documents": documents,
        "limit": request.limit,
        "offset": request.offset,
        "next_offset": request.offset + len(documents),
    }


@app.post("/search", tags=["search"])
def vector_search(request: SearchRequest, service: Service):
    return service.search(
        request.collection,
        request.text,
        request.filter,
        request.top_k,
        request.search_config,
    )


@app.post("/hybrid-search", tags=["search"])
def hybrid_search(request: HybridSearchRequest, service: Service):
    return service.hybrid_search(
        request.collection,
        request.text,
        request.filter,
        request.top_k,
        request.dense_weight,
        request.sparse_weight,
    )


@app.post("/full-text-search", tags=["search"])
def full_text_search(request: FullTextSearchRequest, service: Service):
    return service.full_text_search(
        request.collection, request.text, request.filter, request.top_k
    )


@app.post("/indexes", tags=["indexes"], status_code=201)
def create_index(request: IndexCreate, service: Service):
    return service.create_index(request)


@app.get("/indexes", tags=["indexes"])
def list_indexes(
    service: Service, collection: str = Query(default="demo_documents")
):
    return {"indexes": service.list_indexes(collection)}


@app.delete("/indexes", tags=["indexes"])
def drop_index(request: IndexDrop, service: Service):
    return service.drop_index(request.collection, request.index_name)


@app.post("/partitions", tags=["partitions"], status_code=201)
def create_partition(request: PartitionCreate, service: Service):
    return service.create_partition(request.collection, request.partition_name)


@app.get("/partitions", tags=["partitions"])
def list_partitions(
    service: Service, collection: str = Query(default="demo_documents")
):
    return {"partitions": service.list_partitions(collection)}


@app.delete("/partitions", tags=["partitions"])
def drop_partition(request: PartitionDrop, service: Service):
    return service.drop_partition(request.collection, request.partition_name)
