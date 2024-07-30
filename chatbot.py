from typing import List, Optional

from openai import OpenAI

from db import VectorDb, ChromaDb, DATASET_ID, NAME as DB_NAME


MODEL_NAME = 'gpt-4o-mini'
NUM_RETRIEVAL_RESULTS = 50
MODEL_PROMPT_TEMPLATE = """You are a biomedical scientist.
Answer the question below using the information contained after the heading "Context:". Also quote the most relevant parts of the context which you used to answer the question. If you don't know the answer, say "I do not know".

Context:

{context}
"""

class BiomedRAG:
    def __init__(self, db: VectorDb):
        self.db = db
        self.client = OpenAI()

    def _query_llm(self, prompt: str) -> List[str]:
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

    def query(self, user_query: str) -> List[str]:
        context_docs = self.db.query(user_query, NUM_RETRIEVAL_RESULTS)
        model_prompt = MODEL_PROMPT_TEMPLATE.format( 
            context='\n\n'.join(context_docs)
        )
        return self._query_llm(model_prompt)


if __name__ == '__main__':
    db = ChromaDb(DB_NAME, dataset_id=DATASET_ID)
    rag = BiomedRAG(db)
    print('Type a biomedical question\n')
    try:
        while True:
            user_query = input('> ')
            if user_query:
                responses = rag.query(user_query)
                print('\n\n'.join(responses))
    except (KeyboardInterrupt, EOFError):
        print()
