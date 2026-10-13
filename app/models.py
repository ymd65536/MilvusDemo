from typing import Any, Literal

from pydantic import BaseModel, Field


class CollectionCreate(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    dimension: int = Field(default=384, ge=2, le=32768)
    enable_bm25: bool = True


class CollectionAction(BaseModel):
    action: Literal["load", "release"]


class Document(BaseModel):
    id: str
    text: str
    category: str
    rating: float = 0
    year: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentsWrite(BaseModel):
    collection: str = "demo_documents"
    documents: list[Document] = Field(min_length=1, max_length=1000)
    partition_name: str | None = None


class DeleteRequest(BaseModel):
    collection: str = "demo_documents"
    ids: list[str] = Field(min_length=1)


class GetRequest(BaseModel):
    collection: str = "demo_documents"
    ids: list[str] = Field(min_length=1)


class QueryRequest(BaseModel):
    collection: str = "demo_documents"
    filter: str = ""
    output_fields: list[str] = Field(
        default_factory=lambda: ["id", "text", "category", "rating", "year", "metadata"]
    )
    limit: int = Field(default=10, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class SearchRequest(BaseModel):
    collection: str = "demo_documents"
    text: str
    filter: str = ""
    top_k: int = Field(default=5, ge=1, le=100)
    search_config: Literal["auto", "hnsw"] = "auto"


class HybridSearchRequest(SearchRequest):
    dense_weight: float = Field(default=0.7, ge=0, le=1)
    sparse_weight: float = Field(default=0.3, ge=0, le=1)


class FullTextSearchRequest(BaseModel):
    collection: str = "demo_documents"
    text: str
    filter: str = ""
    top_k: int = Field(default=5, ge=1, le=100)


class IndexCreate(BaseModel):
    collection: str = "demo_documents"
    field_name: str = "embedding"
    index_name: str = "embedding_index"
    index_type: Literal["AUTOINDEX", "HNSW", "IVF_FLAT"] = "AUTOINDEX"
    metric_type: Literal["COSINE", "L2", "IP", "BM25"] = "COSINE"
    params: dict[str, Any] = Field(default_factory=dict)


class IndexDrop(BaseModel):
    collection: str = "demo_documents"
    index_name: str


class PartitionCreate(BaseModel):
    collection: str = "demo_documents"
    partition_name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")


class PartitionDrop(PartitionCreate):
    pass
