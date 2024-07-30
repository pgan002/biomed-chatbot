from itertools import chain
from typing import List, Optional

import nltk


nltk.download('punkt')


def extract_document_id(row: dict) -> str:
    return row['file'].split('/')[1].lstrip('PCM').rstrip('.txt')


def extract_document(row: dict) -> str:
    return row['text']


def clean_and_chunk(text: str) -> List[str]:
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

