from src.query.query_transformer import (
    normalize_query, 
    summarize_query
)

from src.query.query_router import route_query
from src.retrieval.vector_search import seach_movies

from src.retrieval.hybrid_search import(
    keyword_search,
    hybrid_search
)

from src.retrieval.reranker import(
    rerank_document
)

from src.retrieval.compressor import compress_movie_contexts

def run_query_pipeline(query: str) -> dict:

    normalized_query = normalize_query(query)

    transformed_query = summarize_query(normalized_query)

    route = route_query(normalized_query)

    if route == "keyword":
        movies = keyword_search(
            transformed_query,
            top_k=20
        )

    elif route == "hybrid":
        movies = hybrid_search(
            transformed_query,
            top_k=20
        )

    else:
        movies = seach_movies(transformed_query, top_k=20)

    movies = rerank_document(transformed_query, movies, 5)
    compressed_movies = compress_movie_contexts(query=transformed_query, movies=movies)
    return {
        "original_query": query,
        "tranformed_query": transformed_query,
        "route": route,
        "movies":compressed_movies,
    }


if __name__ == "__main__":
    query = input("What kind of movie you are looking for?")

    result = run_query_pipeline(query)

    print(f"\nOriginal Query:{result['original_query']}")
    print(f"\nTransformed Query:{result['tranformed_query']}")
    print(f"\nRoute:{result['route']}")
    print("\nRetrived movies:")

    for rank, movie in enumerate(result["movies"], start=1):
        title = movie["metadata"].get("title", "Unknow title")
        print(f"{rank}. {title}")

        if "distance" in movie:
            print(f"  Vector distance : {movie['distance']:.4f}")

        if "bm25_score" in movie:
            print(f"  BM25 score: {movie['bm25_score']:.4f}")

        if "rrf_score" in movie:
            print(f"  RRF score: {movie['rrf_score']:.5f}")

        print(f"  Genres: {movie['metadata'].get('genres','Unknown')}")