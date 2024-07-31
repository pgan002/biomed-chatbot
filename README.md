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

If you want to use the GloVe embedding (fast) instead of the deep (better) one, download the pre-trained GloVe embedding file from [13].

## Storing documents

To download and ingest data into the database, run

```
python db.py
```

There are two options for embedding that can be set by coding.
- `all-MiniLM-L6-v2`
-- expected to be better
-- ingests about 1 document/s
- `Glove`
-- 10 ingests about 10 documents/s
-- Uses about 3.5G memory during ingestion.

See the discussion under section "Embedding".

If ingestion is interrupted and restarted, it knows from which input document to resume.


## Querying the chatbot

After (enough) documents are ingested, start the chatbot by running:
```
python chatbot.py
```
### Example 1

Prompt:

```
You are a biomedical scientist.
Below is some information and a question at the end. Use it to answer the question and quote the relevant parts. If none of the information is relevant, ignore it and say "I cannot find relevant information.". Do not make up an answer.
```

`CONTEXT_NUM_DOCS` = 50
`MAX_DISTANCE` = 0.5

Interaction:

```
python chatbot.py 
[nltk_data] Downloading package punkt to /home/philip/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
Vector DB: ChromaDb
Collection: "pubmed-noncommercial"
Embedding: ONNXMiniLM_L6_V2
Type a biomedical question

> What is the effect of sleep on memory?
Several lines of evidence indicate that sleep plays a crucial role in memory consolidation. "Sleep promotes memory consolidation" and "post-learning sleep enhances memory recall." Studies specifically highlight that sleep helps to protect memories from interference and supports their organization, enhancing performance and creativity. 

During sleep, particularly during non-rapid eye movement (NREM) sleep, memories are thought to be reactivated and transferred from the hippocampus to cortical networks, which strengthens and solidifies the memory traces established during learning. "A possible mechanism by which memories could be gradually strengthened is their reactivation during sleep." 

Furthermore, sleep quality and the amount of different sleep stages can significantly affect memory performance. It has been noted that "the amount of SWS (slow-wave sleep) might be particularly important for consolidation of declarative information during sleep." 

Overall, poor sleep has been associated with cognitive dysfunctions, including impairments in attention and memory, highlighting the importance of adequate sleep for effective cognitive functioning.
```

### Example 2

Prompt:

```
You are a biomedical scientist.
Below is some information and a question at the end. Some of it may be relevant to the question. If so, use it to answer the question and quote the relevant parts. If none of the information is relevant, ignore it and say "I cannot find relevant information.". If you know the answer without any of the information, start your answer with "From my general knowledge:". But if you do not know, say "I do not know" and do not make up an answer.
```

`CONTEXT_NUM_DOCS` = 50
`MAX_DISTANCE` = 0.5

Interaction:

```
python chatbot.py 
> What is the effect of sleep on memory?
From my general knowledge: Sleep plays a crucial role in memory consolidation, which is the process of transforming new information into stable long-term memories. During sleep, particularly during slow-wave sleep (SWS) and rapid eye movement (REM) sleep, the brain reactivates and reorganizes memories, facilitating their transfer from the hippocampus to the neocortex for long-term storage. This reactivation enhances the neural connections associated with newly learned information, making it easier to recall later.

Furthermore, lacking adequate sleep can significantly impair cognitive functions, particularly those related to memory. Sleep deprivation has been shown to negatively affect attention, learning, and memory retention, especially for tasks that are dependent on the hippocampus. Overall, sufficient high-quality sleep is crucial for both the retention and enhancement of various types of memory, including declarative and procedural memory.
```

### Example 3

Prompt:

```
You are a biomedical scientist.
Below is some information and a question at the end. Use it to answer the question and quote the relevant parts. If none of the information is relevant, ignore it and say "I cannot find relevant information.". Do not make up an answer.
```

`CONTEXT_NUM_DOCS` = 10
`MAX_DISTANCE` = 0.9

Interaction:

```
 What is the effect of sleep on memory?
Sleep plays a crucial role in memory consolidation, which is the process by which newly acquired information is transformed into a stable and long-lasting memory. Several key points from the information provided highlight this relationship:

1. **Memory Consolidation and Sleep**: "Several lines of evidence indicate that sleep promotes memory consolidation." This suggests that sleep enhances the ability to retain and recall learned information.

2. **Neuronal Activity During Sleep**: During sleep, particularly in the non-rapid eye movement (NREM) phase, there is reactivation of memory traces, which aids in the transfer of memories from the hippocampus to neocortex for long-term storage. "A possible mechanism by which memories could be gradually strengthened is their reactivation during sleep."

3. **Benefits of Sleep on Memory Performance**: Lack of sleep is associated with memory impairments, especially in tasks reliant on the hippocampus. "Many studies have examined the consequences of lack of sleep on memory, reporting that sleep-deprived animals and humans show memory impairments in hippocampal-dependent but not hippocampal-independent tasks."

4. **Sleep Stages**: Specific sleep stages are associated with different types of memory consolidation. "SWS (slow-wave sleep) has been linked primarily to learning and memory, particularly declarative memory." This indicates that SWS is particularly beneficial for retaining factual information and events.

5. **Targeted Memory Reactivation**: Studies have shown that external cues presented during sleep can improve memory consolidation. "Triggering reactivation processes during sleep by re-exposure to associated memory cues (targeted memory reactivation [TMR]) has been shown to improve memory consolidation."

6. **Effects of Sleep Distribution**: The distribution of sleep, such as incorporating naps, can also impact cognitive performance positively. "This suggests that under conditions of chronic sleep restriction, a split sleep schedule may optimize the cognitive and neurophysiological functions that underpin some aspects of learning."

In summary, sleep is essential for effective memory consolidation and retention, with specific sleep stages playing pivotal roles in these processes. Insufficient sleep can lead to cognitive impairments, particularly affecting memory tasks dependent on the hippocampus.
```

# Data

As suggested in the task description, we use the PubMed articles with a non-commercial license: `TaylorAI/pubmed_noncommercial`, ~60GB in text.
It contains 1,438,313 articles.

Most of the articles are formatted with separate header, body and reference sections. Only the body contains biomedical information. The publication year and author names contained in the header section can be useful for answering questions, but is difficult to parse using simple tools and too expensive and slow using a language model. The references can be useful in chain-of-thought reasoning but this would be difficult.


# Cleaning and chunking

When possible, we extract the article bodies and chunk them into paragraphs. Paragraphs and sentences are more semantically coherent than fixed-size chunks, even if those overlap. Sentences may contain too little context. The optimum chunk size depends on the embedding type, because with simple embeddings such as Glove, large chunks would become too general to be useful. I did not tune the think size because it would require a lot of work and because I cannot evaluate retrieved results (I do not know biomedicine).

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

## Number of results

ChromaDB's query() method has a parameter for this. I set a default value without tuning.

## Maximum distance

ChromaDb's query() method does not offer filtering based on distance but we do it after words. The default value is set without tuning.


# Language model

We start with OpenAI's `gpt-4o-mini`. It is the cheapest and smallest among the best models available today. Context size 128K tokens. Pricing: $0.150 /1M input tokens, $0.600 /1M output tokens. Cheap enough for development and demo.

The cheapest and almost as good would be a locally-run `Llama3` model.


# TODO

- Document an example for "What is the effect of sleep on memory?"
- Tune retrieved set size
- Tune max distance for retrieval
- Split only large chunks?
- Easily configurable LLM, DB and embedding function?
- Use Llama3 LLM locally
- Logging instead of print()
- Unit tests


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
