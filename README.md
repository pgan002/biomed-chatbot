# Data

As suggested in the task description, we use the PubMed articles with a non-commercial license: TaylorAI/pubmed_noncommercial, ~60GB in text.
It contains 1,438,313 articles.

Most of the articles are formatted with separate header, body and reference sections. Only the body contains biomedical information. The publication year and author names contained in the header section can be useful for answering questions, but is difficult to parse using simple tools and too expensive and slow using a language model. The references can be useful in chain-of-thought reasoning but this would be difficult.


# Cleaning and chunking

When possible, we extract the article bodies and chunk them into paragraphs. Paragraphs and sentences are more semantically coherent than fixed-size chunks, even if those overlap. Sentences may contain too little context.

About 1000 of the 50,000 documents are not formatted into clear sections or do not have such sections. For these, we use the whole document as a chunk.

In future, we can implement recursive chunk splitting and joining based on size, into sentences or string slices. We could use a chunking library such as Unstructured[6]. But it seems this exercise is to do our own chunking.

The first data file's chunks add up to 1291381114 chars (696 chars/chunk). At about 4 chars/token[4], that is 174 tokens/chunk. Stats for the whole dataset:
-- Total chars: 33,458,664,792
-- Chunks: 44,315,395
-- Tokens/chunk: 189


# Vector DB

We use Chroma because it is simple to use and sufficient for demo purposes.

Consideraions:
- Milvus was fast[1] but PgVector is too[3]. But PgVector requires writing SQL queries in Python.
- Milvus supports many ANN algos[2], but we do not know how to evaluate nor have time
- Milvus and Weaviate support queries using GraphQL and gRPC, monitoring and logging, batch processing[2] but those are not important for this exercise
- Qdrant is apparently very fast with high precision but low recall[1]
- Vespa seems to have a smaller community
- We could use a library such as `FAISS`. That might be faster and would allow us to use a framework like HuggingFace. But the task seems to ask for a full-fledged DB.
- OpenAI's Assistants API provides a vector store which performs well[7], but for now it stores up to 10,000 files


# Embedding

To avoid costs, start with an embedding running locally, namely Sentence Transformers's `all-MiniLM-L6-v2` (ChromaDB's default).

There are many options for embedding models. ChromaDB has some predefined ones and supports a custom function that could run locally.[8] There are rankings on various benchmarks[9]. However, those are large models that will take long to embed the data. Local models tend to be faster than web-hosted models due to network latency[10].
all-MiniLM-L6-v2  has a good balance between speed and quality. Word embedding models like Glove are fastest.[10]


Using OpenAI, the cheapest embedding is `text-embedding-3-small`: 62,500 pages/$, 8191 tokens, $0.02 /1M tokens[5]. Embedding the whole data would cost about $18. We could halve the cost using batching but that would take up to 24 hours.

# Model

We start with OpenAI's `gpt-4o-mini`. It is affordable, small and intelligent. Context size 128K tokens. Pricing: $0.150 /1M input tokens, $0.600 /1M output tokens.


# TODO

✓ Create a Github or Gitlab project
✓ Install vector DB
✓ Download data
✓ Clean data
- Ingest data
✓ Pass query to DB
✓ Pass retrieved context to model
- Tune retrieved set size
- Split large chunks
- Factor our chatbor class
- Document usage
- Code and try GloVe embedding


# References

[1] https://ann-benchmarks.com
[2] https://zackproser.com/blog/vector-databases-compared#performance-and-benchmarking
[3] https://www.timescale.com/blog/pgvector-is-now-as-fast-as-pinecone-at-75-less-cost/
[4] https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them 
[5] https://openai.com/api/pricing/
[6] https://unstructured.io/platform
[7] https://medium.com/@zilliz_learn/openai-rag-vs-your-customized-rag-which-one-is-better-4c65a7c6317b
[8] https://docs.trychroma.com/guides/embeddings#custom-embedding-functions
[9] https://huggingface.co/spaces/mteb/leaderboard
[10] https://huggingface.co/blog/mteb

