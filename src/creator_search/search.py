from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import time
from opensearchpy import OpenSearch
from .constraints import Constraints
from .features import build_feature_vector, explain_linear
from .rerank import LinearReranker, load_reranker

def build_os_query(description: str, c: Constraints) -> Dict[str, Any]:
    filters = []
    if c.min_followers > 0:
        filters.append({"range": {"follower_count": {"gte": c.min_followers}}})
    if c.country:
        filters.append({"term": {"location.country": c.country}})
    if c.city:
        filters.append({"term": {"location.city": c.city}})
    if c.languages:
        filters.append({"terms": {"language": c.languages}})

    text = {
        "multi_match": {
            "query": description,
            "fields": ["bio^1.0", "keywords^2.0", "verticals^1.5", "recent_text^2.5"],
            "type": "best_fields",
        }
    }

    return {"query": {"bool": {"must": [text], "filter": filters}}}

def retrieve_candidates(
    client: OpenSearch,
    index: str,
    description: str,
    constraints: Constraints,
    candidate_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    t0 = time.perf_counter()
    body = build_os_query(description, constraints)
    res = client.search(index=index, body=body, size=candidate_k)
    hits = list(res["hits"]["hits"])
    t1 = time.perf_counter()
    return hits, {"t_retrieve_ms": (t1 - t0) * 1000.0}

def rerank_hits(
    hits: List[Dict[str, Any]],
    request_meta: Dict[str, Any],
    top_k: int,
    reranker: LinearReranker,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    t0 = time.perf_counter()
    out = []
    for h in hits:
        feats = build_feature_vector(h, request_meta=request_meta)
        s = reranker.score(feats)
        exp = explain_linear(feats, reranker.weights)
        src = h.get("_source") or {}
        out.append({
            "creator_id": src.get("creator_id"),
            "final_score": s,
            "bm25_score": float(h.get("_score") or 0.0),
            "explain": exp,
        })
    out.sort(key=lambda x: x["final_score"], reverse=True)
    t1 = time.perf_counter()
    return out[:max(top_k, 0)], {"t_rerank_ms": (t1 - t0) * 1000.0}

def search_topk(
    client: OpenSearch,
    index: str,
    request_obj: Dict[str, Any],
    constraints: Constraints,
    candidate_k: int,
    k: int,
    reranker_path: Optional[str] = None,
) -> Dict[str, Any]:
    desc = str(request_obj.get("description") or "")
    reranker = load_reranker(reranker_path)

    request_meta = {
        "languages": constraints.languages,
        "country": constraints.country,
    }

    hits, t_retrieve = retrieve_candidates(client, index, desc, constraints, candidate_k)
    ranked, t_rerank = rerank_hits(hits, request_meta=request_meta, top_k=k, reranker=reranker)

    return {
        "top_k": ranked,
        "profiling": {**t_retrieve, **t_rerank},
        "candidate_k": candidate_k,
        "k": k,
    }
