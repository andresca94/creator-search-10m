from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

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


def dcg_exponential(gains: List[float]) -> float:
    s = 0.0
    for i, g in enumerate(gains):
        s += (2.0 ** float(g) - 1.0) / math.log2(i + 2.0)
    return s


def ndcg_at_k(pred_ids: List[str], rel: Dict[str, float], k: int) -> float:
    gains = [float(rel.get(cid, 0.0)) for cid in pred_ids[:k]]
    ideal = sorted([float(v) for v in rel.values()], reverse=True)[:k]
    denom = dcg_exponential(ideal)
    return 0.0 if denom <= 0.0 else dcg_exponential(gains) / denom


def run_eval(
    client: OpenSearch,
    index: str,
    evalset_path: Path,
    candidate_k: int,
    k: int,
    relevance_threshold: float,
    use_gold_as_rel: bool,
    reranker_path: str | None,
    ce_model: str | None,
    ce_rerank_k: int,
    ce_batch_size: int,
    ce_max_length: int,
    ce_alpha: float,
) -> Dict[str, Any]:
    n = 0
    p_sum = 0.0
    r_sum = 0.0
    ndcg_sum = 0.0

    for line in evalset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ex = json.loads(line)
        req = ex["request"]

        constraints = parse_constraints(req)
        res = search_topk(
            client=client,
            index=index,
            request_obj=req,
            constraints=constraints,
            candidate_k=candidate_k,
            k=k,
            reranker_path=reranker_path,
            ce_model=ce_model,
            ce_rerank_k=ce_rerank_k,
            ce_batch_size=ce_batch_size,
            ce_max_length=ce_max_length,
            ce_alpha=ce_alpha,
        )
        pred = [r["creator_id"] for r in res.get("top_k", []) if r.get("creator_id")]

        gold = ex.get("gold") or []

        if use_gold_as_rel:
            gold_set = set(gold)
            hits = sum(1 for cid in pred[:k] if cid in gold_set)
            prec = hits / float(k) if k > 0 else 0.0
            rec = hits / float(len(gold_set)) if gold_set else 0.0
            rel = ex.get("relevance") or {cid: 1.0 for cid in gold}
            nd = ndcg_at_k(pred, rel, k)
        else:
            rel = ex.get("relevance_graded") or ex.get("relevance") or {cid: 1.0 for cid in gold}
            rel_set = {cid for cid, g in rel.items() if float(g) >= float(relevance_threshold)}
            hits = sum(1 for cid in pred[:k] if cid in rel_set)
            prec = hits / float(k) if k > 0 else 0.0
            rec = hits / float(len(rel_set)) if rel_set else 0.0
            nd = ndcg_at_k(pred, rel, k)

        p_sum += prec
        r_sum += rec
        ndcg_sum += nd
        n += 1

    return {
        "Precision@k": (p_sum / n) if n else 0.0,
        "Recall@k": (r_sum / n) if n else 0.0,
        "NDCG@k": (ndcg_sum / n) if n else 0.0,
        "n_queries": n,
        "k": k,
        "candidate_k": candidate_k,
        "index": index,
        "evalset": str(evalset_path),
        "relevance_threshold": float(relevance_threshold),
        "use_gold_as_rel": bool(use_gold_as_rel),
        "reranker_path": reranker_path,
        "ce_model": ce_model,
        "ce_rerank_k": int(ce_rerank_k) if ce_model else 0,
        "ce_batch_size": int(ce_batch_size) if ce_model else 0,
        "ce_max_length": int(ce_max_length) if ce_model else 0,
        "ce_alpha": float(ce_alpha) if ce_model else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=INDEX_DEFAULT)
    ap.add_argument("--evalset", default="evalset_labeled.jsonl")
    ap.add_argument("--candidate-k", type=int, default=2000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--relevance-threshold", type=float, default=2.0)
    ap.add_argument("--use-gold-as-rel", action="store_true")
    ap.add_argument("--reranker", default=None, help="Optional path to reranker_weights.json")

    # NEW: CE
    ap.add_argument("--ce-model", default=None)
    ap.add_argument("--ce-rerank-k", type=int, default=100)
    ap.add_argument("--ce-batch-size", type=int, default=32)
    ap.add_argument("--ce-max-length", type=int, default=256)
    ap.add_argument("--ce-alpha", type=float, default=1.0)

    args = ap.parse_args()

    client = make_client()
    out = run_eval(
        client=client,
        index=args.index,
        evalset_path=Path(args.evalset),
        candidate_k=args.candidate_k,
        k=args.k,
        relevance_threshold=args.relevance_threshold,
        use_gold_as_rel=args.use_gold_as_rel,
        reranker_path=args.reranker,
        ce_model=args.ce_model,
        ce_rerank_k=args.ce_rerank_k,
        ce_batch_size=args.ce_batch_size,
        ce_max_length=args.ce_max_length,
        ce_alpha=args.ce_alpha,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
