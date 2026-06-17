def evaluate(chunker,text,embedder,vector_store_class):
    queries={
        "notice period":
            "90 days",

        "annual leave":
            "20 days",

        "health insurance":
            "health insurance",

        "remote work":
            "three days"
    }
    chunks=chunker.split(text)
    store=vector_store_class()
    for chunk in chunks:
        vector=embedder.embed(
            chunk
        )
        store.add(
            chunk,
            vector
        )
    correct=0
    total=len(queries)
    for query,expected in queries.items():
        query_vector=embedder.embed(
            query
        )
        results=store.search(
            query_vector,
            top_k=1
        )
        retrieved_text=results[0][1]

        if expected.lower() in retrieved_text.lower():
            correct+=1
    accuracy=(
        correct/total
    )*100
    return accuracy