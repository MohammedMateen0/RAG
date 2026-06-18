# RAG From Scratch

A beginner-friendly implementation of the core Retrieval-Augmented Generation (RAG) pipeline without using LangChain, Chroma, Pinecone, or any external vector database.

This project was built as part of Week 14 of the Hyderabad ML Mission 2026 and focuses on understanding how RAG works internally before using higher-level frameworks.

---

# Project Overview

Large Language Models (LLMs) have a major limitation: their knowledge is frozen at training time. They cannot access newly created information or private company documents unless that information is provided during inference.

Retrieval-Augmented Generation (RAG) solves this problem by retrieving relevant information from external documents and supplying it to the LLM as context.

This project implements the retrieval portion of a RAG system from scratch.

Pipeline:

Document → Chunk → Embed → Store → Retrieve

---

# Why RAG?

Traditional LLM workflow:

User Question → LLM → Answer

Problems:

* Knowledge cutoff
* No access to private documents
* Hallucinations
* No citations

RAG workflow:

User Question
↓
Retriever
↓
Relevant Chunks
↓
LLM
↓
Grounded Answer

Benefits:

* Access to private data
* Reduced hallucinations
* More factual responses
* Source attribution

---

# Project Structure

```text
rag_from_scratch/
│
├── main.py
├── chunker.py
├── embedder.py
├── vector_store.py
├── sample_doc.txt
└── README.md
```

---

# Components

## 1. Chunker

Responsible for splitting large documents into smaller pieces.

Example:

```text
Original Document

The notice period is 90 days.
Employees receive health insurance.
Remote work is allowed.
```

After chunking:

```text
Chunk 1
Chunk 2
Chunk 3
```

Smaller chunks improve retrieval efficiency and reduce search complexity.

---

## 2. Embedder

Converts text into numerical vectors.

Example:

```text
"The notice period is 90 days"
```

becomes

```text
[0.24, 0.81, -0.13, ...]
```

Vectors capture semantic meaning, allowing similarity search between documents and user queries.

Two embedding approaches were explored:

### Simple Embedding

A handcrafted vector based on:

* Number of words
* Total character count

Used only for understanding the retrieval pipeline.

### Transformer Embedding

Using:

```python
all-MiniLM-L6-v2
```

from Sentence Transformers.

Provides semantic search capabilities.

---

## 3. Vector Store

Stores document vectors and performs similarity search.

Responsibilities:

* Store embeddings
* Compute cosine similarity
* Rank documents
* Return top-k matches

Example:

```text
Query:
"What is the notice period?"

Retrieved:
"The notice period for employees is 90 days."
```

---

# Retrieval Process

## Step 1

Load document.

```text
sample_doc.txt
```

---

## Step 2

Split into chunks.

```python
chunks = chunker.split(text)
```

---

## Step 3

Generate embeddings.

```python
vector = embedder.embed(chunk)
```

---

## Step 4

Store vectors.

```python
store.add(chunk, vector)
```

---

## Step 5

Embed user query.

```python
query_vector = embedder.embed(query)
```

---

## Step 6

Retrieve most similar chunks.

```python
results = store.search(
    query_vector,
    top_k=2
)
```

---

# Cosine Similarity

Similarity between vectors is calculated using cosine similarity.

Formula:

```text
(A · B)
---------
||A|| ||B||
```

Range:

* 1 = identical
* 0 = unrelated
* -1 = opposite

Higher score means greater semantic similarity.

---

# Example Query

Question:

```text
What is the notice period?
```

Retrieved Result:

```text
The notice period for employees is 90 days.
```

Similarity Score:

```text
0.97
```

---

# Installation

Clone repository:

```bash
git clone <repository-url>
cd rag_from_scratch
```

Install dependencies:

```bash
pip install numpy
pip install sentence-transformers
```

---

# Run Project

```bash
python main.py
```

Expected Output:

```text
Results

Score: 0.9732

The notice period for employees is 90 days.

--------------------------------------------------
```

---

# Learning Outcomes

After completing this project, you should understand:

* Why RAG exists
* Why LLMs hallucinate
* Knowledge cutoff limitations
* Chunking fundamentals
* Embedding generation
* Vector storage
* Similarity search
* Cosine similarity
* Retrieval workflow

---

# Current Limitations

This project intentionally avoids production frameworks.

Limitations:

* No PDF ingestion
* No vector database
* No reranking
* No citations
* No LLM generation stage
* No evaluation metrics

These features will be added in later stages of Week 14.

---

# Future Improvements

Day 2:

* Fixed Chunking
* Sliding Window Chunking
* Recursive Chunking
* Semantic Chunking

Day 3:

* Chroma Vector Database
* Embedding Comparisons

Day 4:

* LangChain RAG Pipeline

Day 5:

* MMR Retrieval
* Reranking
* RAGAS Evaluation

Saturday Project:

Legal Document Assistant

```text
PDF
↓
Chunks
↓
Embeddings
↓
Chroma
↓
Retriever
↓
LLM
↓
Answer + Citations
```

---

# Key Interview Takeaway

RAG exists because LLMs cannot access private or newly created information and may hallucinate when knowledge is missing. RAG retrieves relevant information at inference time and grounds the model's response in factual evidence.

Knowledge = RAG

Behavior/Style = Fine-Tuning
# Day 2: Chunking Strategy Comparison

A Retrieval-Augmented Generation (RAG) system is only as good as its retrieval pipeline. One of the most important factors affecting retrieval quality is chunking.

This project explores multiple chunking strategies and evaluates their impact on retrieval performance using the same document, embedding model, and vector store.

---

# Why Chunking Matters

Documents are usually too large to embed and retrieve efficiently as a single unit.

Chunking breaks documents into smaller pieces that can be embedded and searched independently.

Poor chunking can lead to:

* Lost context
* Broken sentences
* Missed retrievals
* Lower answer quality

Even powerful LLMs cannot answer correctly if the retriever fails to provide the relevant context.

---

# Implemented Chunking Strategies

## Fixed Chunking

Splits text into chunks of a fixed size.

Example:

```text
Chunk 1 → Characters 1–100

Chunk 2 → Characters 101–200

Chunk 3 → Characters 201–300
```

Advantages:

* Simple
* Fast

Disadvantages:

* Can split sentences
* May break semantic meaning

---

## Sentence Chunking

Splits documents using sentence boundaries.

Example:

```text
Sentence 1

Sentence 2

Sentence 3
```

Advantages:

* Preserves semantic meaning
* Better retrieval quality

Disadvantages:

* Uneven chunk sizes

---

## Overlap Chunking

Creates overlapping chunks to preserve information near chunk boundaries.

Example:

```text
Chunk 1
[1-100]

Chunk 2
[81-180]
```

Advantages:

* Better context preservation
* Reduces retrieval failures

Disadvantages:

* Increased storage requirements
* Duplicate information

---

# Evaluation Methodology

The same document was indexed using each chunking strategy.

Test Queries:

```text
What is the notice period?

How many annual leave days are provided?

Who receives health insurance?

Is remote work allowed?
```

For each query:

1. Retrieve Top-1 chunk
2. Compare against expected answer
3. Count correct retrievals
4. Compute retrieval accuracy

Formula:

```text
Accuracy =
Correct Retrievals
-------------------
Total Queries
```

---

# Experimental Results

| Chunking Strategy | Accuracy |
| ----------------- | -------- |
| Fixed Chunking    | 75.00%   |
| Sentence Chunking | 100.00%  |
| Overlap Chunking  | 100.00%  |

---

# Result Analysis

Fixed Chunking achieved lower accuracy because some information was split across chunk boundaries.

Sentence Chunking preserved complete semantic units, resulting in perfect retrieval performance on the evaluation set.

Overlap Chunking also achieved perfect retrieval by maintaining context near chunk boundaries through overlapping regions.

These results demonstrate that chunking strategy significantly impacts retrieval quality, even when the embedding model and vector store remain unchanged.

---

# Key Learning

A common misconception is that retrieval quality depends primarily on the embedding model or the LLM.

In practice:

```text
Bad Chunking
        ↓
Bad Retrieval
        ↓
Bad Context
        ↓
Bad Answer
```

Most real-world RAG failures occur before generation, making chunking one of the most critical stages of the pipeline.

---

# Files Added in Day 2

```text
rag_from_scratch/
│
├── chunkers/
│   ├── fixed_chunker.py
│   ├── sentence_chunker.py
│   └── overlap_chunker.py
│
├── evaluation/
│   └── evaluate.py
│
├── experiments/
│   └── compare_chunking.py
```

# Day 3: Embeddings, Vector Databases and Retrieval Systems

After exploring chunking strategies in Day 2, the next step is understanding how documents are represented and retrieved.

This phase introduces embedding models, vector databases, dense retrieval, sparse retrieval, hybrid retrieval, and production retrieval architectures.

---

# Why Retrieval Matters

A Retrieval-Augmented Generation (RAG) system depends on retrieving the correct context before generation begins.

Even the most capable LLM cannot answer correctly if the retriever fails to return the relevant information.

Retrieval quality depends primarily on:

1. Chunking Strategy
2. Embedding Model
3. Retrieval Method
4. Reranking

---

# Dense Retrieval

Dense retrieval uses embedding models to convert text into dense numerical vectors.

Example:

```text
Question:
How many vacation days are available?

Document:
Employees receive 20 days of annual leave.
```

Although the words differ, the meanings are similar.

Dense retrieval captures semantic relationships and retrieves relevant content based on meaning rather than exact wording.

Workflow:

```text
Document
 ↓
Embedding Model
 ↓
Vector

Query
 ↓
Embedding Model
 ↓
Vector

Cosine Similarity
 ↓
Top-K Chunks
```

Advantages:

* Semantic understanding
* Handles paraphrasing
* Strong natural language search

Limitations:

* Can miss rare identifiers
* Can struggle with exact keyword matching

---

# Sparse Retrieval

Sparse retrieval relies on keyword matching instead of semantic similarity.

Algorithms:

* TF-IDF
* BM25

Example:

```text
Query:
Clause 8.2

Document:
Clause 8.2 covers employee termination.
```

Sparse retrieval performs extremely well when exact keywords appear in the document.

Advantages:

* Excellent keyword matching
* Handles legal clauses
* Handles product IDs
* Handles error codes

Limitations:

* Cannot understand semantic meaning
* Misses paraphrased queries

---

# Hybrid Retrieval

Hybrid retrieval combines dense and sparse retrieval.

Workflow:

```text
Dense Search
      +
Sparse Search
      ↓
Score Fusion
      ↓
Top Results
```

Benefits:

* Semantic understanding
* Exact keyword matching
* Higher recall
* Higher precision

Modern production RAG systems commonly use hybrid retrieval because it combines the strengths of both approaches.

---

# Embedding Models

Embeddings convert text into vector representations.

A vector captures semantic information that allows similarity search between queries and documents.

Example:

```text
Employees receive annual leave.
```

becomes:

```text
[0.21, 0.84, -0.13, ...]
```

Similar meanings produce nearby vectors in embedding space.

---

## all-MiniLM-L6-v2

Used in this project.

Characteristics:

* 384 dimensions
* Fast inference
* Lightweight
* Beginner friendly

Suitable for:

* Learning RAG
* Small projects
* Local experimentation

---

## BAAI/bge-m3

Open-source production embedding model.

Characteristics:

* Multilingual
* High retrieval quality
* Supports dense and sparse retrieval

Suitable for:

* Production RAG systems
* Enterprise search

---

## text-embedding-3-large

Commercial embedding model.

Characteristics:

* 3072 dimensions
* Excellent retrieval performance
* Strong multilingual capabilities

Suitable for:

* Large-scale production systems

---

# Embedding Dimensions

Embedding dimension represents the number of values in a vector.

Examples:

```text
384
768
1536
3072
```

Trade-off:

```text
Higher Dimension
        ↓
Better Representation
        ↓
More Storage
        ↓
Higher Cost
```

---

# Vector Databases

Embedding vectors must be stored and searched efficiently.

Vector databases provide fast similarity search for large collections of embeddings.

---

## FAISS

Facebook AI Similarity Search.

Characteristics:

* In-memory vector search
* Extremely fast
* Open source
* Local execution

Best for:

* Research
* Experiments
* Prototyping

Limitations:

* No metadata management
* No persistence layer
* Not a complete database

---

## Chroma

Open-source vector database.

Characteristics:

* Persistent storage
* Metadata support
* Local or cloud deployment
* Easy integration

Best for:

* Learning RAG
* Personal projects
* Local document assistants

This project uses Chroma as the primary vector database.

---

## Pinecone

Managed vector database service.

Characteristics:

* Serverless
* Production ready
* Automatic scaling
* Managed infrastructure

Best for:

* Enterprise deployments
* Large-scale applications

---

## pgvector

Vector extension for PostgreSQL.

Characteristics:

* Store vectors inside PostgreSQL
* SQL support
* No separate vector database

Best for:

* Existing PostgreSQL systems
* Small and medium production applications

---

# Reranking

Retrieval systems typically return the Top-K most similar chunks.

However, the highest similarity score is not always the most relevant document.

Reranking introduces a second-stage model.

Workflow:

```text
Query
 ↓
Retriever
 ↓
Top 20 Chunks
 ↓
Cross Encoder
 ↓
Top 5 Chunks
 ↓
LLM
```

The reranker scores query-document pairs and reorders results based on relevance.

Benefits:

* Higher precision
* Better context quality
* Reduced hallucinations

---

# Project Enhancements

Day 3 introduces:

```text
Dense Retrieval
Sparse Retrieval
Hybrid Retrieval
Chroma Vector Database
FAISS Experimentation
Retrieval Comparison
```

New project structure:

```text
rag_from_scratch/
│
├── retrieval/
│   ├── dense_retriever.py
│   ├── sparse_retriever.py
│   └── hybrid_retriever.py
│
├── vectorstore/
│   ├── chroma_store.py
│   └── faiss_store.py
│
├── experiments/
│   └── compare_retrieval.py
```

---

# Retrieval Comparison

Three retrieval strategies were implemented and evaluated.

Methods:

* Dense Retrieval
* Sparse Retrieval (BM25)
* Hybrid Retrieval

Evaluation focused on retrieving the most relevant chunk for a set of predefined queries.

Example:

```text
Dense Retrieval:
100%

Sparse Retrieval:
75%

Hybrid Retrieval:
100%
```

Results may vary depending on dataset and chunking strategy.

---

# Key Learning

A production RAG system is not simply:

```text
Documents
 ↓
LLM
```

Instead:

```text
Documents
 ↓
Chunking
 ↓
Embeddings
 ↓
Vector Database
 ↓
Retriever
 ↓
Reranker
 ↓
LLM
```

Retrieval quality often has a greater impact on final answer quality than the choice of LLM.

---

# Interview Takeaways

### What is dense retrieval?

Semantic retrieval using embeddings and vector similarity.

### What is sparse retrieval?

Keyword-based retrieval using algorithms such as BM25 and TF-IDF.

### Why use hybrid retrieval?

It combines semantic understanding and exact keyword matching.

### Difference between FAISS and Chroma?

FAISS is a vector search library, while Chroma is a full vector database with storage and metadata support.

### What is reranking?

A second-stage retrieval process that rescoring retrieved documents to improve relevance before generation.

### Why is retrieval important?

Because the LLM can only answer using the context it receives. Poor retrieval leads directly to poor answers.
