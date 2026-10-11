from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME
)

load_dotenv()

chroma_client = chromadb.PersistentClient(
    path = str(CHROMA_DIR)
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

def seach_movies(
    query:str,
    top_k: int = 5,
) -> list[dict]:

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be atleast 1")

    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas","distances"]
    )

    movies = []

    documents = results["documents"][0] or []
    metadatas = results["metadatas"][0] or []
    distances = results["distances"][0] or []
    ids = results["ids"][0] or []

    for movie_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances
    ):
        movies.append(
            {
                "id":movie_id,
                "document":document,
                "metadata":metadata or {},
                "distance":distance
            }
        )

    return movies

if __name__ == "__main__":
    user_query = input("What kind of movie are you looking for?")

    matching_movies = seach_movies(
        query= user_query,
        top_k=10
    )

    print("\nTop matching movies:\n")

    for rank, movie in enumerate(matching_movies, start=1):
        print(f"{rank}. {movie['metadata'].get('title', 'Unknown title')}")
        print(f"  Genres: {movie['metadata'].get('genres','Unknown')}")
        print(f"  Distance : {movie['distance']:.4f}")
        print(f"  Details: {movie['document']}")
        print()