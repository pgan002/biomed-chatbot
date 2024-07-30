import sys
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

    def load(self, file_path: str = None):
        if file_path is None:
            file_path - self.file_path
        print(f'Loading {self.__class__.__name__} word embedding map from file {file_path}')
        with open(file_path, encoding='utf-8') as f:
            for line1 in f:
                n_dims = len(line1.split())
                break
            f.seek(0)
            for i, line in enumerate(f):
                values = line.split()
                if i % 10000 == 0:
                    print(f'\r{i:,}: {values[0]}' + '' * 10, end='')
                start_dim = 1 + len(values) - n_dims
                vector = numpy.asarray(values[start_dim:], 'float32')
                self.dict[values[0]] = vector
        print('\nDone')

    def load_clean_store(self, file_path: str):
        ofn = str(file_path) + '-clean'
        if file_path is None:
            file_path - self.file_path
        print(f'Cleaning {self.__class__.__name__} word embedding map from file {file_path}, storing in file {ofn}')
        with open(file_path, encoding='utf-8') as f:
            for line1 in f:
                n_dims = len(line1.split())
                break
            f.seek(0)
            with open(ofn, 'w', encoding='utf-8') as of:
                for i, line in enumerate(f):
                    values = line.split()
                    if i % 100_000 == 0:
                        print(f'\r{i:,}: {values[0]}' + ' ' * 10, end='')
                    start_dim = 1 + len(values) - n_dims
                    of.write(' '.join([values[0], *values[start_dim:]]) + '\n')
        print('\nDone')
        
    def check(self, file_path = None):
        if file_path is None:
            file_path - self.file_path
        print(f'Checking {self.__class__.__name__} word embedding map file {file_path}')
        with open(file_path, encoding='utf-8') as f:
            for line1 in f:
                n_dims = len(line1.split())
                break
            f.seek(0)
            for i, line in enumerate(f):
                values = line.split()
                if i % 100_000 == 0:
                    print(f'\r{i:,}: {values[0]}' + ' ' * 10, end='')
                diff = len(values) - n_dims
                if diff > 0:
                    print(f'\r{i:,}: {diff} extra values', file=sys.stderr)
                    print('    ' + ' '.join(values[:extra + 1]), file=sys.stderr)
                elif diff < 0:
                    print(f'\r{i:,}: {-diff} missing values', file=sys.stderr)
                    print('    ' + , file=sys.stderr)
                value_errors = ''
                for j, v in enumerate(values[1:]):
                    try:
                        float(v)
                    except ValueError:
                        value_errors += f'{1 + j}:{v} '
                if value_errors:
                    print('    ' + value_errors, file=sys.stderr)

    def embed_word(self, word: str) -> numpy.array:
        return self.dict[word]
    
    def embed_text(self, text: str) -> numpy.array:
        return numpy.mean([self.embed_word(word) for word in text.split()])

