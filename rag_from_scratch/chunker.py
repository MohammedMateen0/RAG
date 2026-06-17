class Chunker:
    def __init__(self,chunk_size=100):
        self.chunk_size=chunk_size
    def split(self,text):
        chunks=[]
        for i in range(0,len(text),self.chunk_size):
            chunks.append(
                text[i:i+self.chunk_size]
            )
        return chunks
    
