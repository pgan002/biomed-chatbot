from pathlib import Path

from chromadb.api.types import (
    Document,
    Documents,
    EmbeddingFunction,
    Embedding,
    Embeddings
)
from gensim.models import KeyedVectors


NAME = 'glove.840B.300d'
FILE_NAME_TEMPLATE = '{name}.txt'
URL_TEMPLATE = 'https://downloads.cs.stanford.edu/nlp/data/{name}.zip'


def get_url(name: str) -> str:
    return URL_TEMPLATE.format(name=name)


def get_file_path(name: str, dir_path: str = 'data') -> str:
    fn = FILE_NAME_TEMPLATE.format(name=name)
    return str(Path(dir_path, fn))


class GloveEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, file_path: str = get_file_path(NAME)):
        self.model = KeyedVectors.load_word2vec_format(file_path)

    def embed_word(self, word: str) -> Embedding:
        return self.model.get_vector(word).tolist()
    
    def embed_text(self, text: Document) -> Embedding:
        return numpy.mean([self.model.get_vector(word) for word in text.split()])
    
    def __call__(self, input_: Documents) -> Embeddings:
        """Embed the input documents."""
        return [self.embed_text(text) for text in input_]
