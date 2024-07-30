#pip install numpy
import numpy


EMBEDDING_FILE_PATH = 'glove.840B.300d.txt'


class GloVeEmbedding:
    def __init__(self, embedding_file_path: str):
        self.embeddings_dict = None
        self.embedding_file_path = embedding_file_path
    
    def load_embeddings(self):
        embeddings_dict = {}
        with open(self.embedding_file_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                vector = numpy.asarray(values[1:], 'float32')
                self.embeddings_dict[values[0]] = vector
    
    def embed_word(word: str) -> numpy.array:
        return self.embeddings_dict[word]
    
    def embed_text(text: str) -> numpy.array:
        return numpy.mean(map(self.embed_word, text.split()))




from chromadb.api.types import (
    Documents,
    EmbeddingFunction,
    Embeddings
)


class GloveEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(
            self,
            my_ef_param: str
    ):
        """Initialize the embedding function."""

    def __call__(self, input: Documents) -> Embeddings:
        """Embed the input documents."""
        return self._my_ef(input)
