from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class LabelingPlan:
    n_requests: int
    pairs_per_request: int
    cost_per_label_usd: float
    labels_per_pair: int = 1

    def total_pairs(self) -> int:
        return self.n_requests * self.pairs_per_request

    def total_labels(self) -> int:
        return self.total_pairs() * self.labels_per_pair

    def total_cost(self) -> float:
        return self.total_labels() * self.cost_per_label_usd

def estimate_labeling(
    target_requests: int = 2000,
    pairs_per_request: int = 20,
    cost_per_label_usd: float = 0.08,
) -> Dict[str, float]:
    plan = LabelingPlan(target_requests, pairs_per_request, cost_per_label_usd)
    return {
        "n_requests": float(plan.n_requests),
        "pairs_per_request": float(plan.pairs_per_request),
        "total_pairs": float(plan.total_pairs()),
        "total_labels": float(plan.total_labels()),
        "cost_per_label_usd": float(plan.cost_per_label_usd),
        "estimated_total_cost_usd": float(plan.total_cost()),
    }
