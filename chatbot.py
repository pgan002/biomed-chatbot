from abc import ABC, abstractmethod
from typing import List, Optional

from openai import OpenAI

from db import AbstractVectorDb, ChromaDb, DATASET_ID, NAME as DB_NAME


MODEL_NAME = 'gpt-4o-mini'
NUM_RETRIEVAL_RESULTS = 50
MODEL_PROMPT_TEMPLATE = """You are a biomedical scientist.
Below is some information and a question at the end. Use it to answer the question and quote the relevant parts. If none of the information is relevant, ignore it and say "I cannot find relevant information.". Do not make up an answer.

Information:

{context}
"""

class BiomedRAG(ABC):
    @abstractmethod
    def _query_model(self, prompt: str) -> List[str]:
        pass
    
    def query(
        self, 
        user_query: str, 
        num_retrieval_results: int = NUM_RETRIEVAL_RESULTS
    ) -> List[str]:
        context_docs = self.db.query(user_query, num_retrieval_results)
        model_prompt = MODEL_PROMPT_TEMPLATE.format( 
            context='\n\n'.join(context_docs)
        )
        return self._query_model(model_prompt)


class OpenAiBiomedRAG(BiomedRAG):
    def __init__(self, db: AbstractVectorDb):
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
    db = ChromaDb()
    rag = OpenAiBiomedRAG(db)
    print('Type a biomedical question\n')
    try:
        while True:
            user_query = input('> ')
            if user_query:
                responses = rag.query(user_query)
                print('\n\n'.join(responses))
    except (KeyboardInterrupt, EOFError):
        print()
