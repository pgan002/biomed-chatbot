import sys
from pathlib import Path

import nltk
import numpy
from chromadb.api.types import (
    Document,
    Documents,
    EmbeddingFunction,
    Embedding,
    Embeddings
)
from gensim.models import KeyedVectors

from utils import download_unzip


FILE_PATH = 'data/glove.840B.300d.txt'
URL_TEMPLATE = 'https://downloads.cs.stanford.edu/nlp/data/{name}.zip'


def get_url(file_path: str) -> str:
    return URL_TEMPLATE.format(name=Path(file_path).stem())


def check_header(file_path: str) -> bool:
    with open(file_path) as f:
        for line in f:
            break
    values = line.split()
    return len(values) == 2 and values[0].isdigit() and values[1].isdigit()


class GloveEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, file_path: str = None):
        self.file_path = file_path
        self._model = None
    
    def load(self):
        if self.file_path is None:
            self.file_path = FILE_PATH
            if not Path(FILE_PATH).exists():
                download_unzip(get_url(self.file_path))
        no_header = not check_header(self.file_path)
        print(f'{self.__class__.__name__}: load embedding file "{self.file_path}"')
        self._model = KeyedVectors.load_word2vec_format(self.file_path, no_header=no_header)

    @property
    def model(self) -> KeyedVectors:
        if self._model is None:
            self.load()
        return self._model

    def embed_text(self, text: Document) -> Embedding:
        tokens = nltk.tokenize.word_tokenize(text)
        return self.model.get_mean_vector(tokens).tolist()
    
    def __call__(self, input_: Documents) -> Embeddings:
        """Embed the input documents."""
        return [self.embed_text(text) for text in input_]
