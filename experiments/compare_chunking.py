from chunkers.fixed_chunker import (
    Chunker
)
from chunkers.sentence_chunker import (
    SentenceChunker
)
from chunkers.overlap_chanker import (
    OverlapChunker
)
from embeddings.embedder import (
    SimpleEmbedder
)
from vectorstore.vector_store import (
    VectorStore
)
from evaluation.evaluate import (
    evaluate
)
with open(
    "data/sample_doc.txt",
    encoding="utf-8"
    ) as file:
    text=file.read()
embedder=SimpleEmbedder()

chunkers={
    "Fixed Chunking":
    Chunker(
        chunk_size=100
    ),
    "Sentence Chunking":
    SentenceChunker(),
    "Overlap Chunking":
    OverlapChunker(
        chunk_size=100,
        overlap=20
    )

}
print(
    "\nChunking Comparison\n"
)

for name, chunker in chunkers.items():

    accuracy = evaluate(
        chunker,
        text,
        embedder,
        VectorStore
    )

    print(
        f"{name}: "
        f"{accuracy:.2f}%"
    )