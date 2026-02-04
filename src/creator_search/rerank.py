from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import json

DEFAULT_WEIGHTS = {
    "text_01": 0.55,
    "eng_01": 0.25,
    "recency_01": 0.15,
    "lang_match": 0.05,
    "geo_match": 0.05,
    "safety_penalty": -0.60,
}

@dataclass(frozen=True)
class LinearReranker:
    weights: Dict[str, float]
    bias: float = 0.0

    def score(self, feats: Dict[str, float]) -> float:
        s = self.bias
        for k, w in self.weights.items():
            s += w * float(feats.get(k, 0.0))
        return float(s)

def load_reranker(path: Optional[str]) -> LinearReranker:
    if not path:
        return LinearReranker(DEFAULT_WEIGHTS, 0.0)
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return LinearReranker(weights=dict(obj.get("weights") or DEFAULT_WEIGHTS), bias=float(obj.get("bias") or 0.0))
