import re

def normalize_query(query: str) -> str:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    return re.sub(r"\s+"," ", query).strip()

def summarize_query(query: str) -> str:

    query = normalize_query(query)

    filter_patterns = [
        r"\bI am looking for\b",
        r"\bI'm looking for\b",
        r"\bI want to watch\b",
        r"\bI would like to watch\b",
        r"\bCan you recommend\b",
        r"\bCould you recommend\b",
        r"\bPlease recommend\b",
        r"\bI would like\b",
        r"\bI want\b",
        r"\bmovies that are\b",
        r"\bmovies which are\b",
    ]

    for pattern in filter_patterns:
        query = re.sub(
            pattern,
            " ",
            query,
            flags=re.IGNORECASE
        )

    query = re.sub(r"\s", " ", query).strip(" ,.!?")

    return query

if __name__ =="__main__":

   example = (
       "Can you recommend movies that are funny "
       "and about space exploration?"
   )

   print("Original:",example)
   print("Normalized:", normalize_query(example))
   print("Search Query:", summarize_query(example))