import re
from src.retrieval.reranker import reranker_model

def split_into_sentences(text: str) -> list[dict] :
    if not text:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    return [s.strip() for s in sentences if s.strip()]


def compress_movie_contexts(query:str, movies:list[dict], thershold: float = 0.0) -> list[dict]:

    if not query or not query.strip():
        raise ValueError('query cannot be empty')

    compressed_movies = []

    for movie in movies:
        doc = movie.get("document","")
        sentences = split_into_sentences(doc)

        if not sentences:
            compressed_movies.append(movie)
            continue

        pairs = [[query, sentence] for sentence in sentences]

        scores = reranker_model.predict(pairs)
        scored_sentences = []

        for sentence, score in zip(sentences, scores):
            if score >= thershold:
                scored_sentences.append({"sentence":sentence, "score" : float(score)})

        scored_sentences = sorted(
            scored_sentences,
            key= lambda x: x["score"],
            reverse= True
        )

        top_sentences = [item["sentence"] for item in scored_sentences[:3]]
        compressed_doc = " ".join(top_sentences)

        compressed_movie = movie.copy()
        compressed_movie["original_document"] = doc
        compressed_movie["document"] = compressed_doc if compressed_doc else doc
        compressed_movie["compressed"] = True
        compressed_movies.append(compressed_movie)

    return compressed_movies


