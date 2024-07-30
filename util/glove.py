from pathlib import Path
from typing import Union

import numpy

from util.downloader import download_unzip


NAME = 'glove.840B.300d'
FILE_NAME_TEMPLATE = '{name}.txt'
URL_TEMPLATE = 'https://downloads.cs.stanford.edu/nlp/data/{name}.zip'


def get_url(name: str) -> str:
    return URL_TEMPLATE.format(name=name)


def get_file_path(name: str, dir_path: str = 'data') -> str:
    fn = FILE_NAME_TEMPLATE.format(name=name)
    return Path(dir_path, fn)


class Glove:
    def __init__(
        self, 
        file_path: Union[Path, str] = get_file_path(NAME)
    ):
        self.dict = {}
        self.file_path = file_path

    def download(self):
        download_unzip(
            get_url(NAME),
            self.file_path
        )

    def load(self):
        print(f'Loading {self.__class__.__name__} word embedding map from file {self.file_path}')
        with open(self.file_path, encoding='utf-8') as f:
            for line1 in f:
                n_dims = len(line1.split())
                break
            f.seek(0)
            for i, line in enumerate(f):
                values = line.split()
                if i % 1000 == 0:
                    print(f'\r{i:,}: {values[0]}', end='')
                start_dim = 1 + len(values) - n_dims
                vector = numpy.asarray(values[start_dim:], 'float32')
                self.dict[values[0]] = vector
        print('Done')

    def embed_word(word: str) -> numpy.array:
        return self.dict[word]
    
    def embed_text(text: str) -> numpy.array:
        return numpy.mean(map(self.embed_word, text.split()))

