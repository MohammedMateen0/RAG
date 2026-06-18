from chunker import Chunker
from embedder import SimpleEmbedder
from vector_store import VectorStore
from vectorstore.chroma_store import (
    ChromaStore
)

text=open("rag_from_scratch/sample_doc.txt",
          encoding='utf-8').read()
chunker=Chunker(100)

chunks=chunker.split(text)

embedder=SimpleEmbedder()
store=VectorStore()
for chunk in chunks:
    vector=embedder.embed(
        chunk
    )

    store.add(
        chunk,vector
    )
query="What is the notice period?"
query_vector=embedder.embed(query)

results=store.search(
    query_vector,
    top_k=2
)

print("\nResults\n")

for score, doc in results:

    print(
        f"Score: {score:.4f}"
    )

    print(doc)
    print("-"*50)


with open(
    "data/sample_doc.txt",
    encoding="utf-8"
) as file:
    text = file.read()

chunks = text.split("\n")
store = ChromaStore()

embeddings = [
    embedder.embed(chunk)
    for chunk
    in chunks
]
store.add(
    ids=[
        str(i)
        for i
        in range(
            len(chunks)
        )
    ],
    documents=chunks,
    embeddings=embeddings
)

query = input(
    "Question: "
)

query_embedding = (
    embedder.embed(
        query
    )
)

results = store.search(
    query_embedding
)

print(results)