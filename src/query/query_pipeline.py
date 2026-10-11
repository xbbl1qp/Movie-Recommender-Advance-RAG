from src.query.query_transformer import (
    normalize_query, 
    summarize_query
)

from src.query.query_router import route_query
from src.retrieval.vector_search import seach_movies

def run_query_pipeline(query: str) -> dict:

    normalized_query = normalize_query(query)

    transformed_query = summarize_query(normalized_query)

    route = route_query(normalized_query)

    movies = seach_movies(transformed_query, top_k=5)

    return {
        "original_query": query,
        "tranformed_query": transformed_query,
        "route": route,
        "movies":movies,
    }


if __name__ == "__main__":
    query = input("What kind of movie you are looking for?")

    result = run_query_pipeline(query)

    print(f"\nOriginal Query:{result['original_query']}")
    print(f"\nTransformed Query:{result['tranformed_query']}")
    print(f"\nRoute:{result['route']}")
    print("\nRetrived movies:")

    for index, movie in enumerate(result["movies"], start=1):
        title = movie["metadata"].get("title", "Unknow title")
        print(f"{index}. {title} (distance={movie['distance']:.4f})")
        