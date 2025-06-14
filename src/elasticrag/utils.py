from typing import List


def rrf(*queries, k: int = 60) -> List[tuple]:
    """Reciprocal Rank Fusion algorithm"""
    ranks = [{d[0]: i + 1 for i, d in enumerate(q)} for q in queries]
    result = {}
    for rank in ranks:
        for d in rank.keys():
            result[d] = (result[d] if d in result else 0) + 1.0 / (k + rank[d])
    return sorted(result.items(), key=lambda kv: kv[1], reverse=True)
