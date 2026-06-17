class OverlapChunker:
    def __init__(self,chunk_size=100,overlap=20):
        self.chunk_size=chunk_size
        self.overlap=overlap
    def split(self,text):
        chunks=[]
        step=(
            self.chunk_size-self.overlap
        )
        for i in range(0,len(text),step):
            chunks.append(
                text[i:i+self.chunk_size]
            )
        return chunks