import re
class SentenceChunker:
    def split(self,text):
        return re.split(
            r'(?<=[.!?])\s+',
            text
        )