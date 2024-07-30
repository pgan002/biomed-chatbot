from pathlib import Path
import numpy


EMBEDDING_NAME = 'glove.840B.300d'
EMBEDDING_FILE_NAME_TEMPLATE = '{embedding_name}.txt'
EMBEDDING_URL_TEMPLATE = \
    'https://downloads.cs.stanford.edu/nlp/data/{embedding_name}.zip'


def get_url(embedding_name: str) -> str:
    return EMBEDDING_URL_TEMPLATE.format(embedding_name=embedding_name)


def get_file_path(embedding_name: str, dir_path: str) -> str:
    return Path.join(
        dir_path,
        EMBEDDING_FILE_NAME_TEMPLATE.format(embedding_name=embedding_name)
    )


class Glove:
    def __init__(
        self, 
        embedding_file_path: str = get_file_path(EMBEDDING_NAME)
    ):
        self.embeddings_dict = None
        self.embedding_file_path = embedding_file_path

    def download(self):
        download_unzip(
            get_url(EMBEDDING_NAME),
            self.embedding_file_path
        )

    def load(self):
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

