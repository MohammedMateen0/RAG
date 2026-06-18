from rank_bm25 import BM250kapi

class SparseRetriever:
    def __init__(self,documents):
        self.documents=documents
        tokenized=[
            doc.split()
            for doc in documents
        ]

        self.bm25=(
            BM250kapi(tokenized)
        )
    def retrieve(self,query,top_k=3):
        scores=(
            self.bm25.get_score(
                query.split()
            )
        )
        ranked=sorted(
            zip(
                self.documents,
                scores
            ),
            key=lambda x:x[1],
            reverse=True
        )
        return ranked[:top_k]