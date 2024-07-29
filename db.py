import json
import re
from pathlib import Path
from typing import List

import chromadb
from chromadb.utils import embedding_functions


DATASET_ID = 'TaylorAI/pubmed_noncommercial'
VECTOR_DB_NAME = 'pubmed-noncommercial'
CHUNK_MAX_CHARS = 4 * 8191
CHINK_MIN_CHARS = 80
CONTEXT_NUM_DOCS = 15
CHUNK_AVERAGE_NUM_TOKENS = 174


def clean_and_chunk(text: str, doc_ix: int, doc_id: str) -> List[str]:
    if text.count('\\n') / len(text) > 1/200:
        text = text.replace('\\n', '\n')
    parts = text.split('\n==== Body\n')
    if len(parts) >= 2:
        body = parts[1].strip()
        #regex = re.compile(r'==== Body\n(.*)\n=== Refs')
        # Parsing using `re` might be faster, but this is simpler
        body = body.split('\n==== ')[0]
        paras = body.split('\n\n')
        return list(filter(lambda x: len(x) >= CHINK_MIN_CHARS, paras))
    return [text]
    # TODO: Split or join chunks depending on size (min and max chars)


class DB:
    def __init__(self, data_dir_path: str = 'data'):
        self.data_dir_path = data_dir_path
        self.client = chromadb.PersistentClient()
        self.dataset_id = DATASET_ID
    
    def ingest_overwrite(self):
        #dataset = load_dataset(DATASET_ID)
        # We would normally use this, but downloading takes long and 
        # I already downloaded the files using HTTP so use those instead
        if any(VECTOR_DB_NAME == c.name for c in self.client.list_collections()):
            self.client.delete_collection(name=VECTOR_DB_NAME)
        collection = self.client.create_collection(
            name=VECTOR_DB_NAME,
        )
        path = Path(self.data_dir_path)
        tot_len = 0
        num_chunks = 0
        for file_ix, filepath in enumerate(path.glob('*.jsonl')):
            with open(filepath) as f:
                for doc_ix, json_s in enumerate(f):
                    d = json.loads(json_s)
                    doc_id = d['file'].split('/')[1].lstrip('PCM').rstrip('.txt')
                    print(f'File: {file_ix}, doc: {doc_ix} id: {doc_id}', end='\r')
                    text = d['text']
                    chunks = clean_and_chunk(text, doc_ix, doc_id)
                    tot_len += len(''.join(chunks))
                    num_chunks += len(chunks)
                    if chunks:
                        ids = [f'{doc_id}.{i}' for i in range(len(chunks))]
                        collection.add(documents=chunks, ids=ids)
        print()
        print('Total chars:', tot_len)
        print('Chunks:', num_chunks)
        print('Tokens/chunk:', round(tot_len / num_chunks / 4), '(assuming 4 chars/token)')

    def get_or_ingest_collection(self):
        try:
            return self.client.get_collection(name=VECTOR_DB_NAME)
        except ValueError:
            self.ingest_overwrite()
            return self.client.get_collection(name=VECTOR_DB_NAME)
    
    def query(self, text: str, n_results: int):
        collection = self.get_or_ingest_collection()
        return collection.query(query_texts=[text], n_results=n_results)


if __name__ == '__main__':
    db = DB()
    db.ingest_overwrite()
