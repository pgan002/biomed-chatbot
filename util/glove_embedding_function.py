from chromadb.api.types import (
    Documents,
    EmbeddingFunction,
    Embeddings
)

from util.glove import Glove


class GloveEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self):
        self.embedding = Glove()
        self.embedding.load()

    def __call__(self, input_: Documents) -> Embeddings:
        """Embed the input documents."""
        return [self.embedding.embed_text(text) for text in input_]
