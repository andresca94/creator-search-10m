from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import time

from opensearchpy import OpenSearch

from .constraints import Constraints
from .features import build_feature_vector, explain_linear
from .rerank import LinearReranker, load_reranker

from .cross_encoder import (
    CrossEncoderConfig,
    get_cross_encoder,
    rerank_with_cross_encoder,
)


def build_os_query(description: str, c: Constraints) -> Dict[str, Any]:
    filters: List[Dict[str, Any]] = []
    should: List[Dict[str, Any]] = []

    # Hard filter: min_followers only
    if c.min_followers > 0:
        filters.append({"range": {"follower_count": {"gte": c.min_followers}}})

    # Soft boosts (NOT filters)
    if c.country:
        should.append({"term": {"location.country": {"value": c.country, "boost": 1.2}}})
    if c.city:
        should.append({"term": {"location.city": {"value": c.city, "boost": 1.1}}})
    if c.languages:
        should.append({"terms": {"language": c.languages}})

    text = {
        "multi_match": {
            "query": description,
            "fields": ["bio^1.0", "keywords^2.0", "verticals^1.5", "recent_text^2.5"],
            "type": "best_fields",
        }
    }

    return {
        "query": {
            "bool": {
                "must": [text],
                "filter": filters,
                "should": should,
                "minimum_should_match": 0,
            }
        }
    }


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
    hits = list((res.get("hits", {}) or {}).get("hits", []) or [])
    t1 = time.perf_counter()
    return hits, {"t_retrieve_ms": (t1 - t0) * 1000.0}


def rerank_hits_linear(
    hits: List[Dict[str, Any]],
    request_meta: Dict[str, Any],
    reranker: LinearReranker,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for h in hits:
        feats = build_feature_vector(h, request_meta=request_meta)
        s = float(reranker.score(feats))
        exp = explain_linear(feats, reranker.weights)
        src = h.get("_source") or {}
        out.append(
            {
                "creator_id": src.get("creator_id"),
                "final_score": s,
                "bm25_score": float(h.get("_score") or 0.0),
                "explain": exp,
                # keep source for CE stage
                "_source": src,
            }
        )
    out.sort(key=lambda x: float(x["final_score"]), reverse=True)
    return out


def search_topk(
    client: OpenSearch,
    index: str,
    request_obj: Dict[str, Any],
    constraints: Constraints,
    candidate_k: int,
    k: int,
    reranker_path: Optional[str] = None,
    # Cross-encoder options
    ce_model: Optional[str] = None,
    ce_rerank_k: int = 100,
    ce_batch_size: int = 16,
    ce_max_length: int = 256,
    ce_alpha: float = 1.0,
    ce_device: str = "auto",
    ce_num_threads: int = 2,  # CPU only
) -> Dict[str, Any]:
    desc = str(request_obj.get("description") or "")
    reranker = load_reranker(reranker_path)

    request_meta = {
        "languages": constraints.languages,
        "country": constraints.country,
    }

    # Retrieve
    hits, t_retrieve = retrieve_candidates(client, index, desc, constraints, candidate_k)

    # Linear rerank
    t1 = time.perf_counter()
    ranked = rerank_hits_linear(hits, request_meta=request_meta, reranker=reranker)
    t2 = time.perf_counter()

    profiling: Dict[str, float] = {**t_retrieve, "t_linear_ms": (t2 - t1) * 1000.0}

    # Optional CE stage
    if ce_model:
        t3 = time.perf_counter()

        cfg = CrossEncoderConfig(
            model_name=str(ce_model),
            batch_size=int(ce_batch_size),
            max_length=int(ce_max_length),
            device=str(ce_device),
            num_threads=int(ce_num_threads),
            log_every_batches=25,
        )
        ce = get_cross_encoder(cfg)

        ranked = rerank_with_cross_encoder(
            query=desc,
            hits=ranked,
            top_m=int(ce_rerank_k),
            ce=ce,
            alpha=float(ce_alpha),
        )

        t4 = time.perf_counter()
        profiling["t_ce_ms"] = (t4 - t3) * 1000.0
        profiling["ce_rerank_k"] = float(ce_rerank_k)

    top_k = ranked[: max(int(k), 0)]

    return {
        "top_k": top_k,
        "profiling": profiling,
        "candidate_k": int(candidate_k),
        "k": int(k),
        "ce_model": ce_model,
        "ce_rerank_k": int(ce_rerank_k) if ce_model else 0,
    }
