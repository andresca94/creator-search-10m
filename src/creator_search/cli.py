from __future__ import annotations

import argparse
import json
from opensearchpy import OpenSearch

from .constraints import parse_constraints
from .search import search_topk
from .eval import run_offline_eval, benchmark_latency
from .labeling import estimate_labeling

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
    )


def cmd_search(args: argparse.Namespace) -> int:
    with open(args.request, "r", encoding="utf-8") as f:
        req = json.load(f)

    constraints = parse_constraints(req)
    client = make_client()

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
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    # NOTE: run_offline_eval currently calls search_topk without CE.
    # For CE-aware eval use scripts/eval_pipeline.py below.
    client = make_client()
    out = run_offline_eval(client, args.index, args.evalset, args.candidate_k, args.k)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_latency(args: argparse.Namespace) -> int:
    client = make_client()
    out = benchmark_latency(client, args.index, args.evalset, args.candidate_k, args.k, args.n)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_labeling(args: argparse.Namespace) -> int:
    out = estimate_labeling(args.requests, args.pairs_per_request, args.cost_per_label)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("creator-search")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="Retrieve from OpenSearch, rerank (linear + optional CE), return Top-K.")
    s.add_argument("--index", default=INDEX_DEFAULT)
    s.add_argument("--request", required=True)
    s.add_argument("--candidate-k", type=int, default=2000)
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--reranker", default=None, help="Optional path to reranker_weights.json")

    # NEW: CE knobs
    s.add_argument("--ce-model", default=None, help="Optional cross-encoder model name (HF).")
    s.add_argument("--ce-rerank-k", type=int, default=100, help="Rerank top M linear results with CE.")
    s.add_argument("--ce-batch-size", type=int, default=32)
    s.add_argument("--ce-max-length", type=int, default=256)
    s.add_argument("--ce-alpha", type=float, default=1.0, help="Combine score = linear + alpha*ce_score")

    s.set_defaults(fn=cmd_search)

    e = sub.add_parser("eval", help="Offline eval (linear only). Use scripts/eval_pipeline.py for CE eval.")
    e.add_argument("--index", default=INDEX_DEFAULT)
    e.add_argument("--evalset", required=True)
    e.add_argument("--candidate-k", type=int, default=2000)
    e.add_argument("--k", type=int, default=10)
    e.set_defaults(fn=cmd_eval)

    l = sub.add_parser("latency", help="Benchmark latency (linear only).")
    l.add_argument("--index", default=INDEX_DEFAULT)
    l.add_argument("--evalset", required=True)
    l.add_argument("--candidate-k", type=int, default=2000)
    l.add_argument("--k", type=int, default=10)
    l.add_argument("--n", type=int, default=50)
    l.set_defaults(fn=cmd_latency)

    lab = sub.add_parser("labeling", help="Estimate labeling volume + cost")
    lab.add_argument("--requests", type=int, default=2000)
    lab.add_argument("--pairs-per-request", type=int, default=20)
    lab.add_argument("--cost-per-label", type=float, default=0.08)
    lab.set_defaults(fn=cmd_labeling)

    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
