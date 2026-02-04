from __future__ import annotations
import argparse, json
from opensearchpy import OpenSearch
from rich import print as rprint

from .constraints import parse_constraints
from .search import search_topk
from .eval import run_offline_eval, benchmark_latency
from .labeling import estimate_labeling

def cmd_search(args: argparse.Namespace) -> int:
    with open(args.request, "r", encoding="utf-8") as f:
        req = json.load(f)
    c = parse_constraints(req)
    client = OpenSearch(args.opensearch)
    res = search_topk(
        client=client,
        index=args.index,
        request_obj=req,
        constraints=c,
        candidate_k=args.candidate_k,
        k=args.k,
        reranker_path=args.reranker,
    )
    rprint(res)
    return 0

def cmd_eval(args: argparse.Namespace) -> int:
    client = OpenSearch(args.opensearch)
    metrics = run_offline_eval(client, args.index, args.evalset, args.candidate_k, args.k)
    rprint(metrics)
    return 0

def cmd_latency(args: argparse.Namespace) -> int:
    client = OpenSearch(args.opensearch)
    out = benchmark_latency(client, args.index, args.evalset, args.candidate_k, args.k, args.n)
    rprint(out)
    return 0

def cmd_labeling(args: argparse.Namespace) -> int:
    out = estimate_labeling(args.requests, args.pairs_per_request, args.cost_per_label)
    rprint(out)
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("creator-search")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search")
    s.add_argument("--opensearch", default="http://localhost:9200")
    s.add_argument("--index", default="creators_v1")
    s.add_argument("--request", required=True)
    s.add_argument("--candidate-k", type=int, default=5000)
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--reranker", default=None, help="path to reranker_weights.json")
    s.set_defaults(fn=cmd_search)

    e = sub.add_parser("eval")
    e.add_argument("--opensearch", default="http://localhost:9200")
    e.add_argument("--index", default="creators_v1")
    e.add_argument("--evalset", required=True)
    e.add_argument("--candidate-k", type=int, default=5000)
    e.add_argument("--k", type=int, default=10)
    e.set_defaults(fn=cmd_eval)

    l = sub.add_parser("latency")
    l.add_argument("--opensearch", default="http://localhost:9200")
    l.add_argument("--index", default="creators_v1")
    l.add_argument("--evalset", required=True)
    l.add_argument("--candidate-k", type=int, default=5000)
    l.add_argument("--k", type=int, default=10)
    l.add_argument("--n", type=int, default=50)
    l.set_defaults(fn=cmd_latency)

    lab = sub.add_parser("labeling")
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
