from __future__ import annotations
from typing import Dict, List, Tuple
import math, time
import numpy as np
from opensearchpy import OpenSearch
from .constraints import parse_constraints
from .search import search_topk

def precision_at_k(pred: List[str], gold: List[str], k: int) -> float:
    p = pred[:k]
    if not p: return 0.0
    g = set(gold)
    return sum(1 for x in p if x in g) / float(len(p))

def recall_at_k(pred: List[str], gold: List[str], k: int) -> float:
    g = set(gold)
    if not g: return 0.0
    p = set(pred[:k])
    return len(p.intersection(g)) / float(len(g))

def ndcg_at_k(pred: List[str], rel: Dict[str, float], k: int) -> float:
    def dcg(items: List[str]) -> float:
        s = 0.0
        for i, cid in enumerate(items[:k]):
            r = float(rel.get(cid, 0.0))
            s += (2**r - 1.0) / math.log2(i + 2.0)
        return s
    ideal = sorted(rel.items(), key=lambda x: x[1], reverse=True)
    ideal_list = [cid for cid,_ in ideal]
    denom = dcg(ideal_list)
    return 0.0 if denom == 0.0 else dcg(pred) / denom

def run_offline_eval(
    client: OpenSearch,
    index: str,
    evalset_path: str,
    candidate_k: int,
    k: int,
) -> Dict[str, float]:
    # evalset.jsonl lines:
    # {"request": {...}, "gold": ["cr_.."], "relevance": {"cr_..": 3, ...}}
    import json
    ps, rs, ns = [], [], []
    with open(evalset_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            req = ex["request"]
            gold = ex.get("gold") or []
            rel = ex.get("relevance") or {cid: 1.0 for cid in gold}

            c = parse_constraints(req)
            res = search_topk(client, index, req, c, candidate_k=candidate_k, k=k)
            pred = [r["creator_id"] for r in res["top_k"] if r.get("creator_id")]

            ps.append(precision_at_k(pred, gold, k))
            rs.append(recall_at_k(pred, gold, k))
            ns.append(ndcg_at_k(pred, rel, k))

    return {
        f"Precision@{k}": float(np.mean(ps) if ps else 0.0),
        f"Recall@{k}": float(np.mean(rs) if rs else 0.0),
        f"NDCG@{k}": float(np.mean(ns) if ns else 0.0),
    }

def benchmark_latency(
    client: OpenSearch,
    index: str,
    evalset_path: str,
    candidate_k: int,
    k: int,
    n_requests: int = 50,
) -> Dict[str, float]:
    import json
    times = []
    with open(evalset_path, "r", encoding="utf-8") as f:
        lines = [json.loads(x) for x in f.readlines()]
    for ex in lines[:n_requests]:
        req = ex["request"]
        c = parse_constraints(req)
        t0 = time.perf_counter()
        _ = search_topk(client, index, req, c, candidate_k=candidate_k, k=k)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    if not times:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    arr = np.asarray(times, dtype=float)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(np.mean(arr)),
    }
