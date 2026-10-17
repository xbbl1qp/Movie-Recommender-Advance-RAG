import csv
import json
from pathlib import Path
from statistics import mean

from src.retrieval.vector_search import seach_movies
from src.retrieval.hybrid_search import (
    keyword_search,
    hybrid_search
)

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "golden_queries.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "retrieval_evaluation.csv"
)

TOP_K = 5

def load_dataset():
    with open(DATASET_PATH, "r", encoding="UTF-8") as file:
        return json.load(file)

def extract_movie_ids(movies : list) -> list[int]:
    movie_ids = []

    for movie in movies:
        metadata = movie.get("metadata", {})
        movie_id = metadata.get("movie_id")

        if movie_id is None:
            raise ValueError(
                f"Retrieved movie has no movie_id : {movie}"
            )

        movie_ids.append(int(movie_id))

    return movie_ids

def evaluate_retrieval():
    test_cases = load_dataset()

    strategies = {
        "vector" : seach_movies,
        "keyword" : keyword_search,
        "hybrid": hybrid_search
    } 

    rows = []

    for case in test_cases:
        query = case["query"]
        relevant_ids = {
            int(movie_id)
            for movie_id in case["relevant_movie_ids"]
        }

        for strategy_name, search_function in strategies.items() :
            movies = search_function(query, top_k = TOP_K)
            retrieved_ids = extract_movie_ids(movies)

            row = {
                "query" : query,
                "strategy": strategy_name,
                "expected_route": case["expected_route"],
                "result_count": len(retrieved_ids),
                "precision_at_5": precision_at_k(
                    relevant_ids= relevant_ids,
                    retrieved_ids= retrieved_ids,
                    k = TOP_K 
                ),
                "recall_at_5": recall_at_k(
                    retrieved_ids, relevant_ids, TOP_K
                ),
                "mrr": reciprocal_rank(
                    retrieved_ids, relevant_ids
                ),
                "ndcg_at_5": ndcg_at_k(
                    retrieved_ids, relevant_ids, TOP_K
                ),
                "retrieved_movie_ids": retrieved_ids,
            }

            rows.append(row)

            print(
                f"{strategy_name:8} | "
                f"P@5={row['precision_at_5']:.3f} | "
                f"R@5={row['recall_at_5']:.3f} | "
                f"MRR={row['mrr']:.3f} | "
                f"nDCG@5={row['ndcg_at_5']:.3f} | "
                f"{query}"
            )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = list(rows[0].keys()) if rows else []

    with open(
        REPORT_PATH,
        "w",
        newline="",
        encoding="UTF-8"
    ) as file :
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

        print("\n===Average Retrieval Metrics ===")

        for strategy_name in strategies:
            strategy_rows = [
                row for row in rows
                if row['strategy'] == strategy_name
            ]

            if not strategy_rows:
                continue

            print(f"\n{strategy_name.upper()}")

            for metric in (
                "precision_at_5",
                "recall_at_5",
                "mrr",
                "ndcg_at_5"
            ):
                score = mean(
                    row[metric] for row in strategy_rows
                )

                print(f"{metric}:  {score:.4f}")

        print(f"\n CSV report saved to : {REPORT_PATH}")


if __name__ == "__main__":
   evaluate_retrieval()