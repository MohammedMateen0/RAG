import chromadb

class ChromaStore:
    def __init__(self):
        self.client=(
            chromadb.Client()
        )
        self.collection=(
            self.client.create_collection(
                "documents"
            )
        )
    def add(self,ids,documents,embeddings):
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings
        )
    def search(self,query_embedding,top_k=3):
        return self.collection.query(

            query_embedding=[
                query_embedding
            ],
            n_results=top_k
        )