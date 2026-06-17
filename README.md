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

---

# Interview Takeaways

### Why is chunking important?

Chunking directly affects retrieval quality. Poor chunk boundaries can separate related information and reduce retrieval accuracy.

### Why use overlap?

Overlap preserves context near chunk boundaries and reduces the chance of missing important information.

### Why is sentence chunking often better than fixed chunking?

Sentence chunking preserves complete semantic units and avoids breaking information across chunks.

### What did this experiment demonstrate?

Changing only the chunking strategy improved retrieval accuracy from 75% to 100%, highlighting the importance of retrieval engineering in RAG systems.
