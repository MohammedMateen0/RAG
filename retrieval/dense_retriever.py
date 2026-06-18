class DenseRetriver:
    def __init__(self,embedder,store):
        self.embedder=embedder
        self.store=store
    def retrieve(self,query,top_k=3):
        query_vector=(
            self.embedder.embed(
                query
            )
        )
        return self.store.search(
            query_vector,
            top_k
        )
        