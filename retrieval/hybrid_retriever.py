class HybridRetriever:
    def __init__(self,dense,sparse):
        self.dense=dense
        self.sparse=sparse
    def retrieve(self,query):
        dense_results=(
            self.dense.retrieve(
                query
            )
        )
        sparse_results=(
            self.sparse.retrieve(
                query
            )
        )
        return {
            "dense":dense_results,
            "sparse":sparse_results
        }