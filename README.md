# Usage

## Setup

1. Install python3.10
1. Then run:

```
git clone https://github.com/pgan002/chatbot.git
cd chatbot
python3.10 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=<key>
```

replacing the actual key in the last line

If you want to use the GloVe embedding (fast) instead of the LLM one (better), download the pre-trained GloVe embedding file from [13].

## Storing documents

To download and ingest data into the database, run

```
python db.py
```

There are two options for embedding that can be set by coding. See the discussion under section "Embedding".
- The faster one (Glove) uses about 3.5G during ingestion.
- The slower and better (`all-MiniLM-L6-v2`) uses less memory and, we implement auto-resume if ingestion is interrupted. 


## Querying the chatbot

After (enough) documents are ingested, start the chatbot by running:
```
python chatbot.py
```

# Data

As suggested in the task description, we use the PubMed articles with a non-commercial license: `TaylorAI/pubmed_noncommercial`, ~60GB in text.
It contains 1,438,313 articles.

Most of the articles are formatted with separate header, body and reference sections. Only the body contains biomedical information. The publication year and author names contained in the header section can be useful for answering questions, but is difficult to parse using simple tools and too expensive and slow using a language model. The references can be useful in chain-of-thought reasoning but this would be difficult.


# Cleaning and chunking

When possible, we extract the article bodies and chunk them into paragraphs. Paragraphs and sentences are more semantically coherent than fixed-size chunks, even if those overlap. Sentences may contain too little context. It is difficult to compare results of each because I am not an expert in the application domain, and it would require a lot of work.

About 1000 of the 50,000 documents are not formatted into clear sections or do not have such sections. For these, we use the whole document as a chunk.

In future, we can implement recursive chunk splitting and joining based on size, into sentences or string slices. We could use a chunking library such as Unstructured[6]. But it seems this exercise is to do our own chunking.

The first data file's chunks add up to 1291381114 chars (696 chars/chunk). At about 4 chars/token[4], that is 174 tokens/chunk. Stats for the whole dataset:
-- Total chars: 33,458,664,792
-- Chunks: 44,315,395
-- Tokens/chunk: 189


# Vector DB

We use Chroma because it is simple to use and sufficient for demo purposes.

- Milvus was fast[1] but PgVector is too[3]. But PgVector requires writing SQL queries in Python.
- Milvus supports many ANN algos[2], but we do not know how to evaluate nor have time
- Milvus and Weaviate support queries using GraphQL and gRPC, monitoring and logging, batch processing[2] but those are not important for this exercise
- Qdrant is apparently very fast with high precision but low recall[1]
- Vespa seems to have a smaller community
- We could use a library such as `FAISS`. That might be faster and would allow us to use a framework like HuggingFace. But the task seems to ask for a full-fledged DB.
- OpenAI's Assistants API provides a vector store which performs well[7], but for now it stores up to 10,000 files
- Pinecone is proprietary, and I prefer to avoid that.

The main difficulty with storing the documents for retrieval is the time to vectorize and index so much text. Vectorization speed depends on the choice of embedding function, indexing speed on the indexing algorithm in the database (hard-coded). I found anecdotal information and ingestion speed measurements for one or two vector databases, so it is difficult to compare.

# Embedding

There are many possible embedding functions. ChromaDB predefines some and supports a custom ones.[8] Various benchmarks rank embedding functions[9]. Most of those are large models that will take long. Local models tend to be faster than web-hosted models due to network latency[10].

`all-MiniLM-L6-v2`  has a good balance between speed and quality.

Word embedding models like Glove are the fastest.[10] but do not take account of the context of words in the document. Specifically GloVe also ignores word order. Thus retrieval will miss important results and include irrelevant results (low recall and precision).

To avoid costs, I started with an embedding running locally, namely Sentence Transformers's `all-MiniLM-L6-v2` (ChromaDB's default). Embedding progressed at 1 document / s, so embedding the whole dataset would take 16 days. This did not work on my GPU, although other models do use it.

OpenAI's cheapest embedding is `text-embedding-3-small`: 62,500 pages/$, 8191 tokens, $0.02 /1M tokens[5]. Embedding the whole data would cost about $18. Vectorization progressed at about 1 document per second. The cost would be only $9 by batching, that would take up to 24 hours. This is a good option, although I did not use it.

Other AI providers also host embedding models, sometimes cheaper, but no batching.

Instead, I implemented document embedding via averaged Glove word embeddings.


# Retrieval

## Distance metric

Publications find Euclidean distance metric better than cosine distance in ChromaDB[11] and for embeddings in general[12]. This is the default for ChromaDB.

ChromaDb's query() method does not offer filtering based on distance but we do it after words.


# Language model

We start with OpenAI's `gpt-4o-mini`. It is the cheapest and smallest among the best models available today. Context size 128K tokens. Pricing: $0.150 /1M input tokens, $0.600 /1M output tokens. Cheap enough for development and demo.

The cheapest and almost as good would be a locally-run `Llama3` model.


# TODO

- Tune retrieved set size
- Tune max distance for retrieval
- Split only large chunks?
- Easily configurable LLM, DB and embedding function?
- Use Llama3 LLM


# References

[1]: https://ann-benchmarks.com
[2]: https://zackproser.com/blog/vector-databases-compared#performance-and-benchmarking
[3]: https://www.timescale.com/blog/pgvector-is-now-as-fast-as-pinecone-at-75-less-cost/
[4]: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them 
[5]: https://openai.com/api/pricing/
[6]: https://unstructured.io/platform
[7]: https://medium.com/@zilliz_learn/openai-rag-vs-your-customized-rag-which-one-is-better-4c65a7c6317b
[8]: https://docs.trychroma.com/guides/embeddings#custom-embedding-functions
[9]: https://huggingface.co/spaces/mteb/leaderboard
[10]: https://huggingface.co/blog/mteb
[11]: https://medium.com/@stepkurniawan/comparing-similarity-searches-distance-metrics-in-vector-stores-rag-model-f0b3f7532d6f
[12]: https://arxiv.org/pdf/1803.02839
[13]: https://nlp.stanford.edu/data/glove.840B.300d.zip
