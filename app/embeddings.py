from functools import cached_property


class TextEmbedder:
    def __init__(self, model_name: str, dimension: int) -> None:
        self.model_name = model_name
        self.dimension = dimension

    @cached_property
    def model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        result = vectors.tolist()
        if result and len(result[0]) != self.dimension:
            raise ValueError(
                f"Model produces {len(result[0])} dimensions, expected {self.dimension}"
            )
        return result
