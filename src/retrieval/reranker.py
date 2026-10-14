from sentence_transformers import CrossEncoder

from src.config import (
    RERANKER_MODEL_NAME
)

print(f"loading reranker model : {RERANKER_MODEL_NAME} ...")
reranker_model = CrossEncoder(RERANKER_MODEL_NAME)

def rerank_document(query:str, movies: list[dict], top_k = 5) -> list[dict] :

    if not movies:
        return []

    if not query or not query.strip():
        raise ValueError("QUery cannot be empty")

    pairs = []

    for movie in movies:
        title = movie["metadata"].get("title","")
        genres = movie["metadata"].get("genres","")
        doc = movie["document"]
        combined_text = f"Title :{title}. Genres : {genres}. Description: {doc}"
        pairs.append([query,combined_text])

    scores = reranker_model.predict(pairs)

    for i, movie in enumerate(movies):
        movie["cross_encoder_score"] = float(scores[i])

    reranked_movies = sorted(
        movies,
        key= lambda x: x["cross_encoder_score"],
        reverse=True
    )     

    return reranked_movies[:top_k]