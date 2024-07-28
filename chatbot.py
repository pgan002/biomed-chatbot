import os

from openai import OpenAI

from db import DB, CHUNK_AVERAGE_NUM_TOKENS


MODEL_NAME = 'gpt-4o-mini'
OPENAI_API_KEY = ''
#MODEL_CONTEXT_WINDOW = 128_000
NUM_RETREIVAL_RESULTS = 50
MODEL_PROMPT = """You are a biomedical scientist.
Find the answer to the question below using only the context after the question.
In your  answer, also quote the most relevant paragraphs from the context which
you used to answer the question. If you don't know the answer, say "I do not know".
Don't make up an answer."""


if __name__ == '__main__':
    db = DB()
    cleint = OpenAI()
    assistant = client.beta.assistants.create(
        name='PubMed biomedical scientist',
        instructions=MODEL_PROMPT,
        model=MODEL_NAME,
        top_p=0.1,
    )
    thread = client.beta.threads.create()
    print('Type a biomedical question.\n')
    while True:
        try:
            user_query = input(prompt='> ')
            context_results = db.query(user_query, NUM_RETREIVAL_RESULTS)
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role='user',
                content='\n\n'.join([user_query, *context_results])
            )
            run = client.beta.threads.runs.create_and_poll(
                thread_id=thread.id,
                assistant_id=assistant.id,
            )
        except KeyboardInterrupt:
            print()
