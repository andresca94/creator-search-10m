from __future__ import annotations

import argparse
import json
from pathlib import Path
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=INDEX_DEFAULT)
    ap.add_argument("--evalset", default="evalset_labeled.jsonl")
    ap.add_argument("--candidate-k", type=int, default=2000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--relevance-threshold", type=float, default=2.0)
    ap.add_argument("--use-gold-as-rel", action="store_true")
    ap.add_argument("--reranker", default=None)

    # CE knobs
    ap.add_argument("--ce-model", default=None)
    ap.add_argument("--ce-rerank-k", type=int, default=100)
    ap.add_argument("--ce-batch-size", type=int, default=32)
    ap.add_argument("--ce-max-length", type=int, default=256)
    ap.add_argument("--ce-alpha", type=float, default=1.0)

    args = ap.parse_args()

    client = make_client()
    eval_lines = Path(args.evalset).read_text(encoding="utf-8").splitlines()

    rows: List[Tuple[int, float, Dict[str, Any]]] = []

    for line in eval_lines:
        if not line.strip():
            continue
        ex = json.loads(line)
        req = ex["request"]
        q = str(req.get("description") or "")
        constraints = parse_constraints(req)

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
        pred = [r["creator_id"] for r in res.get("top_k", []) if r.get("creator_id")]

        if args.use_gold_as_rel:
            gold = ex.get("gold") or []
            gold_set = set(gold)
            hits = sum(1 for cid in pred[: args.k] if cid in gold_set)
            ndcg = 0.0  # optional; not needed for debug ordering
            miss = [cid for cid in gold if cid not in set(pred[: args.k])]
        else:
            rel = ex.get("relevance_graded") or ex.get("relevance") or {}
            rel_set = {cid for cid, g in rel.items() if float(g) >= float(args.relevance_threshold)}
            hits = sum(1 for cid in pred[: args.k] if cid in rel_set)
            # quick ndcg proxy: average rel in top-k (cheap debug)
            ndcg = sum(float(rel.get(cid, 0.0)) for cid in pred[: args.k]) / float(max(args.k, 1))
            miss = [cid for cid in rel_set if cid not in set(pred[: args.k])][:10]

        rows.append((int(hits), float(ndcg), {"q": q, "pred": pred, "miss": miss}))

    # Worst: lowest hits then lowest ndcg
    rows.sort(key=lambda x: (x[0], x[1]))

    print("WORST by hits then ndcg:\n")
    for hits, nd, info in rows[: args.n]:
        print("---")
        print(f"q: {info['q']}")
        print(f"hits: {hits} ndcg: {nd:.4f}")
        print(f"top_ids: {info['pred'][:args.k]}")
        print(f"miss_relevant: {info['miss']}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
