from __future__ import annotations
from typing import List, Optional

LABEL_PENALTY = {
    "none": 0.0,
    "politics": 0.15,
    "controversy": 0.35,
    "adult": 0.60,
}

def safety_penalty(
    labels: List[str],
    is_spam: bool,
    authenticity_score: Optional[float],
) -> float:
    p = 0.0
    for lab in (labels or []):
        p = max(p, LABEL_PENALTY.get(lab, 0.2))
    if is_spam:
        p = max(p, 0.75)
    if authenticity_score is not None and authenticity_score < 0.6:
        p = max(p, 0.4 + (0.6 - authenticity_score))
    return min(max(p, 0.0), 0.95)
