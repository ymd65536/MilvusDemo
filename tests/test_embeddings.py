import pytest

from app.embeddings import TextEmbedder


class FakeVectors:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeModel:
    def __init__(self, values):
        self.values = values

    def encode(self, texts, normalize_embeddings):
        assert normalize_embeddings is True
        return FakeVectors(self.values)


def test_embedder_checks_configured_dimension():
    embedder = TextEmbedder("unused", 2)
    embedder.__dict__["model"] = FakeModel([[0.1, 0.2]])

    assert embedder.encode(["hello"]) == [[0.1, 0.2]]


def test_embedder_rejects_wrong_model_dimension():
    embedder = TextEmbedder("unused", 3)
    embedder.__dict__["model"] = FakeModel([[0.1, 0.2]])

    with pytest.raises(ValueError, match="expected 3"):
        embedder.encode(["hello"])
