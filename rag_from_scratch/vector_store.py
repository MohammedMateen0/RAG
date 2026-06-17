import numpy as np

class VectorStore:
    def __init__(self):
        self.documents=[]
        self.vectors=[]
    def add(self,text,vector):
        self.documents.append(text)
        self.vectors.append(vector)
    def cosine_similarity(self,a,b):
        return (np.dot(a,b)
                /
                (
                    np.linalg.norm(a)
                    *
                    np.linalg.norm(b)
                )
        )
    def search(self,query_vector,top_k=2):
        scores=[]
        for i,vector in enumerate(self.vectors):
            score=self.cosine_similarity(
                query_vector,
                vector
            )
            scores.append(
                (score,
                self.documents[i])
            )
        scores.sort(
            reverse=True,
            key=lambda x:x[0]
        )
        return scores[:top_k]
