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

from src.generator.generator import generate_recommendation

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

    final_response = generate_recommendation(query, compressed_movies)

    return {
        "original_query": query,
        "tranformed_query": transformed_query,
        "route": route,
        "movies": compressed_movies,
        "response": final_response,
    }


if __name__ == "__main__":
    query = input("What kind of movie you are looking for?")

    print("\n Pipeline running...Please wait for response from model")
    result = run_query_pipeline(query)

    print(f"\nOriginal Query:{result['original_query']}")
    print(f"\nTransformed Query:{result['tranformed_query']}")
    print(f"\nRoute:{result['route']}")

    print("\n" + "="*50)
    print("AI Generated Recommendation:")
    print("="*50)
    print(result["response"])
    print("="*50 + "\n")

    print("Retrived Source Movies")
    for rank, movie in enumerate(result["movies"], start=1):
        title = movie["metadata"].get("title", "Unknown Title")
        print(f"{rank}. {title}, (Genres: {movie['metadata'].get('genres','Unknown')})")