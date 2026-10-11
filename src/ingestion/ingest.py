import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

from src.config import(
    BATCH_SIZE,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    MOVIES_CSV,
    RATINGS_CSV,
    TAGS_CSV
)

def load_movies() -> pd.DataFrame:
    for path in (MOVIES_CSV, RATINGS_CSV, TAGS_CSV):
        if not path.exists():
            raise FileNotFoundError(
                f"Required Dataset not found : {path}\n"
                "Download Movielens ml-latest-small and extract it to data/ml-latest-small/."
            )

    movies = pd.read_csv(MOVIES_CSV)
    ratings = pd.read_csv(RATINGS_CSV)
    tags = pd.read_csv(TAGS_CSV)

    tags_by_movie = (
        tags.groupby("movieId")["tag"]
        .apply(lambda values: ", ".join(sorted({
            str(v).strip() for v in values.dropna() if str(v).strip()
        })))
        .reset_index(name = "tags")
    ) 

    ratings_by_movie = (
        ratings.groupby("movieId")["rating"]
        .agg(average_rating="mean", rating_count= "count")
        .reset_index()
    )

    movies = movies.merge(tags_by_movie, on="movieId", how="left")
    movies = movies.merge(ratings_by_movie, on="movieId", how="left")

    movies["tags"] = movies["tags"].fillna("")
    movies["average_rating"] = movies["average_rating"].fillna(0.0)
    movies["rating_count"] = movies["rating_count"].fillna(0).astype(int)
    movies["genres_clean"] = (
        movies["genres"].fillna("").str.replace("|", ", ", regex=False)
    )

    movies["document"] = (
        "Title: " + movies["title"].fillna("")
        + ". Genres:" + movies["genres_clean"]
        + ". Tags:" + movies["tags"]
        + ". Average MovieLens rating: "
        + movies["average_rating"].round(2).astype(str)
        + ". Number of ratings:" + movies["rating_count"].astype(str)
    )

    return movies

def create_fresh_collection(client):
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception as exc:
        message = str(exc).lower()
        print(f"exception in create_fresh_collection: {message}")
        if "not found" not in message and "does not exist" not in message:
            raise

    return client.create_collection(
        name = COLLECTION_NAME,
        metadata = {"hnsw:space":"cosine"},
    )   

def main() -> None:
    print(f"Loading MovieLens data from : {MOVIES_CSV.parent}")
    movies = load_movies()
    print(f"Movies loaded :{len(movies)}")
    print("Example Docuemnt")
    print(movies.iloc[0]["document"])

    print(f"Loading Embedding Model : {EMBEDDING_MODEL_NAME}")
    encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    CHROMA_DIR.mkdir(parents=True, exist_ok= True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = create_fresh_collection(client)

    for start in range(0, len(movies), BATCH_SIZE):
        batch =  movies.iloc[start : start + BATCH_SIZE]
        documents = batch["document"].tolist()

        embeddings = encoder.encode(
            documents,
            show_progress_bar=False,
            convert_to_numpy=True
        )       
        
        ids = [str(movie_id) for movie_id in batch["movieId"]]
        metadatas = [
            {
                "movie_id":int(row["movieId"]),
                "title":str(row["title"]),
                "genres":str(row["genres_clean"]),
                "tags": str(row["tags"]),
                "average_rating": float(row["average_rating"]),
                "rating_count": int(row["rating_count"])
            }
            for _,row in batch.iterrows()
        ]

        collection.add(
            ids = ids,
            documents = documents,
            embeddings = embeddings.tolist(),
            metadatas = metadatas
        )
        print(f"Indexed {min(start + len(batch), len(movies))}/ {len(movies)} movies")

    print("Ingestion Complete.")
    print(f"Stored records: {collection.count()}")

if __name__ == "__main__":
    main()