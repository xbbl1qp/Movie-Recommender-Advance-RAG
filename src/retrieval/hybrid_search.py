import re

from rank_bm25 import BM25Okapi

from src.retrieval.vector_search import collection, seach_movies

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())

def load_movie_documents() -> list[dict]:
    data = collection.get(
        include=["documents", "metadatas"]
    )
    ids = data["ids"]
    documents = data["documents"] or []
    metadatas = data["metadatas"] or []

    movies = []

    for movie_id, document, metadata in zip(
        ids, documents, metadatas
    ):
        movies.append(
            {
                "id": movie_id,
                "document": document or "",
                "metadata": metadata or {}
            }
        )

    return movies

def keyword_search(
        query: str,
        top_k: int = 5,
) -> list[dict]:

    if not query or not query.strip():
        raise ValueError("Query cannot be Empty")

    if top_k < 1:
         raise ValueError("top_k must be atleast 1")


    movies = load_movie_documents()

    if not movies:
        return []

    tokenized_documents = [
        tokenize(
            f"{movie['metadata'].get('title','')}"
            f"{movie['metadata'].get('genres','')}"
            f"{movie['document']}"
        )
        for movie in movies
    ]

    bm25 = BM25Okapi(tokenized_documents)
    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    ranked_indices  = sorted(
        range(len(scores)),
        key = lambda index:scores[index],
        reverse=True
    )

    results = []

    for index in ranked_indices[:top_k]:
        movie = movies[index]
        results.append(
            {
                **movie,
                "bm25_score":float(scores[index])
            }
        )

    return results 

def reciprocal_rank_fusion(
        ranked_lists = list[list[dict]],
        k: int = 60,
) -> list[dict]:

    fused_scores = {}
    movie_by_id = {}

    for ranked_list in ranked_lists:
        for rank,movie in enumerate(ranked_list, start=1):
            movie_id = movie["id"]

            fused_scores[movie_id] = (
                fused_scores.get(movie_id, 0.0)
                + 1.0 / (k + rank)
            )

            if movie_id not in movie_by_id:
                movie_by_id[movie_id] = movie
            else:
                movie_by_id[movie_id].update(
                    {
                        key : value
                        for key,value in movie.items()
                        if key not in movie_by_id[movie_id]
                    }
                )

    ranked_ids = sorted(
        fused_scores,
        key = fused_scores.get,
        reverse=True
    )

    results = []

    for movie_id in ranked_ids:
        results.append({
           **movie_by_id[movie_id],
           "rrf_score": fused_scores[movie_id],     
        })

    return results

def hybrid_search(
    query: str,
    top_k: int = 5,
    candidate_k: int = 20,
) -> list[dict]:

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1 or candidate_k < 1:
        raise ValueError("top_k and candidate_k must be positive.")

    vector_results = seach_movies(
        query = query,
        top_k=candidate_k
    )

    keyword_results = keyword_search(
        query=query,
        top_k=candidate_k
    )

    fused_results = reciprocal_rank_fusion(
        [vector_results, keyword_results]
    )

    return fused_results[:top_k]

if __name__ == '__main__':
    query = input("Enter a movie query: ")

    print("\n---Keyword search (BM25) ---")

    for movie in keyword_search(query, top_k=5):
        print(
            movie['metadata'].get('title','Unknown'),
            "| BM25:",
            round(movie['bm25_score'], 4)
        )

    print("\n---Hybrid search (RRF) ---")

    for movie in hybrid_search(query, top_k=5):
        print(
            movie['metadata'].get('title', 'unknown'),
            '| RRF:',
            round(movie['rrf_score'], 5),
        )