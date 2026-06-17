from chunker import Chunker
from embedder import SimpleEmbedder
from vector_store import VectorStore

text=open("rag_from_scratch/sample_doc.txt",
          encoding='utf-8').read()
chunker=Chunker(100)

chunks=chunker.split(text)

embeder=SimpleEmbedder()
store=VectorStore()
for chunk in chunks:
    vector=embeder.embed(
        chunk
    )

    store.add(
        chunk,vector
    )
query="What is the notice period?"
query_vector=embeder.embed(query)

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