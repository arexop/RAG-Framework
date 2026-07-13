from semantic_search import documents, model, search_engine


def semantic_search(query, top_k=3):
    # Embed the incoming search query
    query_embedding = model.encode([query])

    # Retrieve the closest matches
    top_k = min(top_k, len(documents))
    distances, indices = search_engine.kneighbors(query_embedding, n_neighbors=top_k)

    print(f"\nQuery: '{query}'")
    print("-" * 50)

    for i in range(top_k):
        doc_idx = indices[0][i]
        # Convert cosine distance to similarity (1 - distance)
        similarity = 1 - distances[0][i]

        print(f"Result {i+1} (Similarity: {similarity:.4f})")
        print(f"Text: {documents[int(doc_idx)][:150]}...\n")
