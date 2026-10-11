import re

KNOWN_EXACT_TITLE_PATTERNS = [
    r"\bthe matrix\b",
    r"\binterstellar\b",
    r"\bback to future\b",
    r"\btitanic\b",
    r"\bthe godfather\b",
]

def route_query(query: str) -> str:
    
    normalized = query.lower().strip()

    if not normalized:
        raise ValueError("Query cannot be empty")

    has_known_title = any(
        re.search(pattern, normalized)
        for pattern in KNOWN_EXACT_TITLE_PATTERNS
    )

    semantic_terms = [
        "similar to",
        "like",
        "feel",
        "vibe",
        "about",
        "recommend",
        "something",
        "movies that",
        "movie where" 
    ]

    has_semantic_intent = any(
        term in normalized
        for term in semantic_terms
    )

    if has_known_title and has_semantic_intent:
        return "hybrid"

    if has_known_title:
        return "keyword"
    
    return "vector"

if __name__ == "__main__":
    examples = [
       "Movies that feel adventurous and uplifting",
       "The Matrix",
       "Funny sci-fi movies similar to The Matrix", 
    ]

    for query in examples:
        print(f"{query!r} -> {route_query(query)}")
  