from abc import ABC, abstractmethod
from typing import List, Optional

from openai import OpenAI

from db import VectorDb, ChromaDb, DATASET_ID, NAME as DB_NAME


MODEL_NAME = 'gpt-4o-mini'
NUM_RETRIEVAL_RESULTS = 50
MODEL_PROMPT_TEMPLATE = """You are a biomedical scientist.
Below some information and a question at the end. The information may be relevant to answering the question. If it is relevant, use it to answer the question and quote the most relevant parts. If none of the information is relevant, say "I cannot find any relevant information" and do not make up an answer.

Information:

{context}
"""

class BiomedRag(ABC):
    @abstractmethod
    def _query_model(self, prompt: str) -> List[str]:
        pass
    
    def query(self, user_query: str) -> List[str]:
        context_docs = self.db.query(user_query, NUM_RETRIEVAL_RESULTS)
        model_prompt = MODEL_PROMPT_TEMPLATE.format( 
            context='\n\n'.join(context_docs)
        )
        return self._query_model(model_prompt)


class OpenAiBiomedRAG(BiomedRag):
    def __init__(self, db: VectorDb):
        self.db = db
        self.client = OpenAI()

    def _query_model(self, prompt: str) -> List[str]:
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    'role': 'system',
                    'content': prompt
                },
                {
                    'role': 'user',
                    'content': user_query
                }
            ]
        )
        return [c.message.content for c in response.choices]


if __name__ == '__main__':
    db = ChromaDb(DB_NAME, dataset_id=DATASET_ID)
    rag = OpenAiBiomedRag(db)
    print('Type a biomedical question\n')
    try:
        while True:
            user_query = input('> ')
            if user_query:
                responses = rag.query(user_query)
                print('\n\n'.join(responses))
    except (KeyboardInterrupt, EOFError):
        print()
