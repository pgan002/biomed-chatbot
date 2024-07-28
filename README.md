== Vector DB

- We could use a library such as FAISS. That would allow us to use a framework like HuggingFace. But the task explicitly asked for and listed DBs
- Chroma is simple to use and sufficient for demo purposes
- Milvus was fast[1] but PgVector is too[3]. But PgVector requires writing SQL queries in Python.
- Milvus supports many ANN algos[2], but we do not know how to evaluate nor have time
- Milvus and Weaviate support queries using GraphQL and gRPC, monitoring and logging, batch processing[2] but those are not important for this exercise
- OpenAI's Assistants API provides a vector store which performs well[7], but for now it stores up to 10,000 files


== Chunking

- We could use a chunking library such as Unstructured[6]. But it seems this exercise is to do our own chunking.
- Paragraphs and sentences are more semantically coherent than fixed-size chunks. Sentences may contain too little context.

== Embedding and model
- The first data file's chunks add up to 1291381114 chars (696 chars/chunk). The dataset contains 28 * 50K PubMed articles. At about 4 chars/token[4], that is 174 tokens/chunk.
-- Stats for the whole dataset:
--- Total chars: 33458664792
--- Chunks: 44315395
--- Tokens/chunk: 189
- OpenAI:
-- The most appropriate embedding `text-embedding-3-small`: 62,500 pages/$, 8191 tokens, $0.020 /1M tokens[5]. It would cost about $18.
-- The most appropriate model is `gpt-4o-mini`: affordable and intelligent small model. Context size 128K tokens. Pricing: $0.150 /1M input tokens, $0.600 /1M output tokens.
- I wish we could parse metadata, such as the year or authors but that is too hard to do using simple tools and too expensive and/or slow using a language model.


== TODO

- Create a Github or Gitlab project
✓ Install vector DB
✓ Download data
- Clean data
- Ingest data
- Pass query to DB
- Pass embedded results to model (?)
- Tune result set size for context window


== References

[1] https://ann-benchmarks.com
[2] https://zackproser.com/blog/vector-databases-compared#performance-and-benchmarking
[3] https://www.timescale.com/blog/pgvector-is-now-as-fast-as-pinecone-at-75-less-cost/
[4] https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them 
[5] https://openai.com/api/pricing/
[6] https://unstructured.io/platform
[7] https://medium.com/@zilliz_learn/openai-rag-vs-your-customized-rag-which-one-is-better-4c65a7c6317b
