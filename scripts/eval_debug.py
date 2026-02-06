from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Tuple

from opensearchpy import OpenSearch

from creator_search.constraints import parse_constraints
from creator_search.search import search_topk

INDEX_DEFAULT = "creators_v1"
ADMIN_USER = "admin"
ADMIN_PASS = "ChangeThis_ToA_StrongPassword_123!"


def make_client() -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        http_auth=(ADMIN_USER, ADMIN_PASS),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
        timeout=60,
        max_retries=2,
        retry_on_timeout=True,
    )


def percentile(xs: List[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=INDEX_DEFAULT)
    ap.add_argument("--evalset", default="evalset_labeled.jsonl")
    ap.add_argument("--candidate-k", type=int, default=2000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--relevance-threshold", type=float, default=2.0)

    ap.add_argument("--reranker", default=None)
    ap.add_argument("--ce-model", default=None)
    ap.add_argument("--ce-rerank-k", type=int, default=100)
    ap.add_argument("--ce-batch-size", type=int, default=32)
    ap.add_argument("--ce-max-length", type=int, default=256)
    ap.add_argument("--ce-alpha", type=float, default=1.0)

    args = ap.parse_args()

    client = make_client()
    eval_lines = Path(args.evalset).read_text(encoding="utf-8").splitlines()

    latencies: List[float] = []
    rows: List[Tuple[int, float, Dict[str, Any]]] = []

    for i, line in enumerate(eval_lines[: args.n]):
        ex = json.loads(line)
        req = ex["request"]
        constraints = parse_constraints(req)

        t0 = perf_counter()
        res = search_topk(
            client=client,
            index=args.index,
            request_obj=req,
            constraints=constraints,
            candidate_k=args.candidate_k,
            k=args.k,
            reranker_path=args.reranker,
            ce_model=args.ce_model,
            ce_rerank_k=args.ce_rerank_k,
            ce_batch_size=args.ce_batch_size,
            ce_max_length=args.ce_max_length,
            ce_alpha=args.ce_alpha,
        )
        t1 = perf_counter()

        latencies.append((t1 - t0) * 1000.0)

        pred = [r["creator_id"] for r in res.get("top_k", []) if r.get("creator_id")]
        rel = ex.get("relevance_graded") or {}
        rel_set = {cid for cid, g in rel.items() if float(g) >= args.relevance_threshold}
        hits = sum(1 for cid in pred[: args.k] if cid in rel_set)

        rows.append((hits, 0.0, {"q": req.get("description"), "pred": pred}))

    print("\nLATENCY (ms):")
    print(f"mean={stats.mean(latencies):.2f}")
    print(f"p50 ={percentile(latencies, 50):.2f}")
    print(f"p95 ={percentile(latencies, 95):.2f}")
    print(f"p99 ={percentile(latencies, 99):.2f}")
    print(f"n   ={len(latencies)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
