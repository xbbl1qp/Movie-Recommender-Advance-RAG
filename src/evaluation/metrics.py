import math

def precision_at_k(
     retrieved_ids: list,
     relevant_ids: set,
     k : int = 5,
) -> float:

    if k<=0:
        raise ValueError("k must be greater than 0")

    top_k = retrieved_ids[:k]

    if not top_k:
        return 0.0

    relevant_count = sum(
        1 for movie_id in top_k
        if movie_id in relevant_ids
    )

    return relevant_count/k


def recall_at_k(
        retrieved_ids : list,
        relevant_ids : set,
        k: int = 5,
) -> float:

    if k < 0 :
        raise ValueError("k must be greater than 0")

    top_k = retrieved_ids[:k]

    relevant_count = sum (
        1 for movie_id in top_k
        if movie_id in relevant_ids
    )

    return relevant_count / len(relevant_ids)


def reciprocal_rank(
        retrieved_ids: list,
        relevant_ids: set,
        k : int = 5,
) -> float:

    for rank,movie_id in enumerate(retrieved_ids, start=1):
        if movie_id in relevant_ids:
            return 1.0/rank

    return 0.0


def ndcg_at_k(
       retrieved_ids: list,
       relevant_ids: set,
       k : int = 5,
) -> float:

    if k<=0:
        raise ValueError("k must be greater than zero")

    top_k = retrieved_ids[:k]

    dcg = sum(
        1.0/math.log2(rank +1)
        for rank,movie_id in enumerate(top_k, start=1)
        if movie_id in relevant_ids
    )

    ideal_relevant_count = min(len(relevant_ids), k)

    ideal_dcg = sum(
        1.0/math.log2(rank +1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    if ideal_dcg == 0.0:
        return 0.0

    return dcg/ideal_dcg