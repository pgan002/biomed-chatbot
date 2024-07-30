from abc import ABC, abstractmethod
from itertools import chain
from typing import List, Optional

import nltk

import chromadb
import datasets


DATASET_ID = 'TaylorAI/pubmed_noncommercial'
#NAME = 'pubmed-noncommercial'
NAME = 'test'
CHINK_MIN_CHARS = 32
CHUNK_MAX_CHARS = 2048
CONTEXT_NUM_DOCS = 15
CHUNK_AVERAGE_NUM_TOKENS = 174
MIN_DISTANCE = 0.9

nltk.download('punkt')


def extract_doc_id(row):
    return row['file'].split('/')[1].lstrip('PCM').rstrip('.txt')


def clean_and_chunk(text: str, doc_ix: int, doc_id: str) -> List[str]:
    n = text.count('\\n')
    if n and len(text) // n > 200:
        text = text.replace('\\n', '\n')
    parts = text.split('\n==== Body\n')
    if len(parts) >= 2:
        body = parts[1].strip()
        #regex = re.compile(r'==== Body\n(.*)\n=== Refs')
        # TODO: Parsing using `re` might be slightly faster
        body = body.split('\n==== ')[0]
        chunks = body.split('\n\n')
    else:
        chunks = [text]
    chunks = list(chain(*(nltk.tokenize.sent_tokenize(c) for c in chunks)))
    chunks = [x for x in chunks if len(x) >= CHINK_MIN_CHARS]
    # TODO: Split or join consecutive chunks depending on size
    return chunks


class VectorDb(ABC):
    def __init__(self, name: str = NAME, dataset_id: str = DATASET_ID):
        print(f'Using {self.__class__.__name__} vector database "{name}"')
        self.name = NAME
        self.dataset_id = dataset_id
    
    @abstractmethod
    def ingest(self):
        print(f'Ingesting dataset "{self.dataset_id}"')
    
    @abstractmethod
    def get_last_doc_ix(self) -> int:
        pass

    @abstractmethod
    def query(self, text: str, n_results: int) -> List[str]:
        pass

    @abstractmethod
    def delete(self):
        print(f'Deleting vector database "{self.name}" if it exists')


class ChromaDb(VectorDb):
    def __init__(self, name: str = NAME, dataset_id: str = DATASET_ID):
        super().__init__(name, dataset_id)
        self.client = chromadb.PersistentClient()
    
    def ingest(self):
        collection = self.client.get_or_create_collection(self.name)
        previously_ingested_ids = set(
            s.split('_')[0] 
            for s in collection.get(include=[])['ids']
        )
        metadata = collection.metadata or {}
        last_doc_ix = metadata.get('last_doc_ix', -1)
        total_chars = metadata.get('total_chars', 0)
        ds = datasets.load_dataset(self.dataset_id)['train']
        try:
            for doc_ix, row in enumerate(ds):
                doc_id = extract_doc_id(row)
                if last_doc_ix is None:
                    if doc_id in previously_ingested_ids:
                        continue
                else:
                    if doc_ix <= last_doc_ix:
                        continue
                print(f'\rIngest "{doc_id}" {doc_ix:,}/{ds.shape[0]:,}', end='')
                text = row['text']
                chunks = clean_and_chunk(text, doc_ix, doc_id)
                if chunks:
                    total_chars += sum(len(c) for c in chunks)
                    ids = [f'{doc_id}_{i}' for i in range(len(chunks))]
                    try:
                        collection.add(documents=chunks, ids=ids)
                    except TypeError as e:
                        print(chunks)
                        raise e
                    metadata = {
                        'last_doc_ix': doc_ix, 
                        'last_doc_id': doc_id, 
                        'total_chars': total_chars
                    }
                    collection.modify(metadata=metadata)
        except KeyboardInterrupt:
            pass
        n_chunks = collection.count()
        toks_per_chunk = round(total_chars / 4 / n_chunks)
        print('Totals:')
        print(f'Chars: {total_chars:,}')
        print(f'Chunks: {n_chunks:,}')
        print(f'Tokens/chunk: {toks_per_chunk:,} (assuming 4 chars/token)')

    def get_last_doc_ix(self) -> int:
        try:
            collection = self.client.get_collection(name=self.name)
        except ValueError:
            return -1
        return (collection.metadata or {}).get('last_doc_ix', -1)
    
    def query(self, text: str, max_results: int, min_distance: float = MIN_DISTANCE) -> List[str]:
        collection = self.client.get_collection(name=self.name)
        response = collection.query(query_texts=[text], n_results=max_results)
        close_docs = [
            s 
            for s, d in zip(response['documents'][0], response['distances'])
            if d <= max_distance
        ]
        return close_results 

    def delete(self):
        super().delete()
        try:
            self.client.delete_collection(name=self.name)
        except ValueError:
            pass


if __name__ == '__main__':
    db = ChromaDb()
    last_doc_ix = db.get_last_doc_ix()
    if last_doc_ix >= 0:
        print(f'Collection {db.name} last ingested document {last_doc_ix}.')
        print('Choose:')
        print(f'[a]ppend, starting at dataset document {1 + last_doc_ix}')
        print(f'[d]elete (replace) the collection')
        if input('(a/d)? ').lower() == 'd':
            db.delete()
    db.ingest()

