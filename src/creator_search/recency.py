from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List
import math

def _days_ago(ts_yyyy_mm_dd: str) -> float:
    try:
        dt = datetime.strptime(ts_yyyy_mm_dd, "%Y-%m-%d")
        return max((datetime.utcnow() - dt).days, 0)
    except Exception:
        return 9999.0

def recency_weight(days: float, half_life_days: float = 14.0) -> float:
    return 0.5 ** (days / half_life_days)

def build_recent_text(posts: List[Dict[str, Any]]) -> str:
    chunks = []
    for p in posts or []:
        days = _days_ago(str(p.get("timestamp", "")))
        w = recency_weight(days)
        base = " ".join([
            str(p.get("caption","") or ""),
            " ".join(p.get("hashtags") or []),
            str(p.get("transcript") or "")
        ]).strip()
        if not base:
            continue
        reps = max(1, int(round(1 + 4 * w)))  # cheap weighting for BM25
        chunks.append((" " + base) * reps)
    return " ".join(chunks)

def compute_recency_score(posts: List[Dict[str, Any]]) -> float:
    ws = []
    for p in posts or []:
        days = _days_ago(str(p.get("timestamp", "")))
        ws.append(recency_weight(days))
    if not ws:
        return 0.0
    return float(1.0 - math.exp(-sum(ws)))  # saturating in (0,1)
