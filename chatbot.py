from openai import OpenAI

from db import DB


MODEL_NAME = 'gpt-4o-mini'
OPENAI_API_KEY = ''
NUM_RETREIVAL_RESULTS = 50
MODEL_PROMPT_TEMPLATE = """You are a biomedical scientist.
Answer the question below using the information contained after the heading "Context:". Also quote the most relevant parts of the context which you used to answer the question. If you don't know the answer, say "I do not know".

Context:

{context}
"""


if __name__ == '__main__':
    db = DB()
    client = OpenAI()
    print('Type a biomedical question\n')
    try:
        while True:
            user_query = input('> ')
            if user_query == '':
                continue
            retreival_result = db.query(user_query, NUM_RETREIVAL_RESULTS)
            context_docs = retreival_result['documents'][0]
            model_prompt = MODEL_PROMPT_TEMPLATE.format( 
                context='\n\n'.join(context_docs)
            )
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        'role': 'system',
                        'content': model_prompt
                    },
                    {
                        'role': 'user',
                        'content': user_query
                    }
                ]
            )
            for c in response.choices:
                print(c.message.content, '\n')
    except (KeyboardInterrupt, EOFError):
        print()
