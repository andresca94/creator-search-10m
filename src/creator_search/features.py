from __future__ import annotations
from typing import Any, Dict, List, Optional
import math
import numpy as np
from .safety import safety_penalty

FEATURE_ORDER = ["text_01","eng_01","recency_01","lang_match","geo_match","safety_penalty"]

def _safe_float(x: Any) -> float:
    try:
        if x is None: return 0.0
        return float(x)
    except Exception:
        return 0.0

def _safe_int(x: Any) -> int:
    try:
        if x is None: return 0
        return int(x)
    except Exception:
        return 0

def engagement_score_01(followers: int, er: float, views: int) -> float:
    f = math.tanh(math.log1p(max(followers, 0)) / 10.0)
    e = math.tanh(max(er, 0.0) * 8.0)
    v = math.tanh(math.log1p(max(views, 0)) / 12.0)
    return float(0.55 * f + 0.35 * e + 0.10 * v)

def build_feature_vector(hit: Dict[str, Any], request_meta: Dict[str, Any]) -> Dict[str, float]:
    src = hit.get("_source") or {}
    bm25 = float(hit.get("_score") or 0.0)
    text_01 = float(1.0 - np.exp(-bm25 / 8.0))

    followers = _safe_int(src.get("follower_count"))
    er = _safe_float(src.get("engagement_rate_30d"))
    views = _safe_int(src.get("avg_views_30d"))

    eng_01 = engagement_score_01(followers, er, views)
    rec_01 = float(src.get("recency_score") or 0.0)

    labels = list(src.get("content_safety_labels") or [])
    is_spam = bool(src.get("is_suspected_spam") or False)
    auth = src.get("authenticity_score")
    auth_f = None if auth is None else _safe_float(auth)

    pen = safety_penalty(labels, is_spam=is_spam, authenticity_score=auth_f)

    req_langs = request_meta.get("languages") or []
    req_country = request_meta.get("country")

    lang_match = 1.0 if (req_langs and src.get("language") in req_langs) else 0.0
    geo_match = 1.0 if (req_country and (src.get("location") or {}).get("country") == req_country) else 0.0

    return {
        "text_01": text_01,
        "eng_01": eng_01,
        "recency_01": rec_01,
        "lang_match": lang_match,
        "geo_match": geo_match,
        "safety_penalty": pen,
    }

def explain_linear(feats: Dict[str, float], weights: Dict[str, float]) -> Dict[str, Any]:
    contrib = {k: float(feats.get(k, 0.0) * weights.get(k, 0.0)) for k in weights.keys()}
    raw = float(sum(contrib.values()))
    pos = sorted([(k,v) for k,v in contrib.items() if v>0], key=lambda x:x[1], reverse=True)[:3]
    neg = sorted([(k,v) for k,v in contrib.items() if v<0], key=lambda x:x[1])[:3]
    return {
        "raw_score": raw,
        "features": feats,
        "contributions": contrib,
        "top_positive": pos,
        "top_negative": neg,
    }
