import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

DATA_DIR = project_path(os.getenv("DATA_DIR", "data/ml-latest-small"))
CHROMA_DIR = project_path(os.getenv("CHROMA_DIR", "chroma_db"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "movies")

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-V2"
)

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))

MOVIES_CSV = DATA_DIR / "movies.csv"
TAGS_CSV = DATA_DIR / "tags.csv"
RATINGS_CSV = DATA_DIR / "ratings.csv"