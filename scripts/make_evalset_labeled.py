from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from opensearchpy import OpenSearch

from creator_search.constraints import parse_constraints
from creator_search.search import build_os_query as build_os_query_runtime
from creator_search.features import build_feature_vector
from creator_search.rerank import load_reranker, LinearReranker
from creator_search.recency import compute_recency_score

from creator_search.cross_encoder import (
    CrossEncoderConfig,
    get_cross_encoder,
    rerank_with_cross_encoder,
)

OPENSEARCH_URL_DEFAULT = "https://localhost:9200"
OPENSEARCH_USER_DEFAULT = "admin"
OPENSEARCH_PASS_DEFAULT = "ChangeThis_ToA_StrongPassword_123!"
INDEX_DEFAULT = "creators_v1"

EXCLUDE_ID_PREFIXES_DEFAULT = ["cr_gold_", "cr_hard_"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def make_client(url: str, user: str, password: str) -> OpenSearch:
    return OpenSearch(
        url,
        http_auth=(user, password),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
        timeout=60,
        max_retries=2,
        retry_on_timeout=True,
    )


def is_excluded_id(cid: str, exclude_prefixes: List[str]) -> bool:
    c = cid or ""
    for p in exclude_prefixes:
        if c.startswith(p):
            return True
    return False


def linear_score_hits(
    hits: List[Dict[str, Any]],
    request_meta: Dict[str, Any],
    reranker: LinearReranker,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for h in hits:
        src = h.get("_source") or {}
        cid = str(src.get("creator_id") or "")
        if not cid:
            continue

        # Ensure recency_score exists for feature extraction
        if "recency_score" not in src:
            src["recency_score"] = compute_recency_score(src.get("recent_posts") or [])

        feats = build_feature_vector(h, request_meta=request_meta)
        s = float(reranker.score(feats))

        out.append(
            {
                "creator_id": cid,
                "linear_score": s,
                "bm25_score": float(h.get("_score") or 0.0),
                "_source": src,
                "_hit": h,
            }
        )

    out.sort(key=lambda x: float(x["linear_score"]), reverse=True)
    return out


def teacher_rerank_with_ce(
    query: str,
    linear_ranked: List[Dict[str, Any]],
    ce_model: str,
    ce_rerank_k: int,
    ce_batch_size: int,
    ce_max_length: int,
    ce_alpha: float,
    ce_device: str,
    ce_num_threads: int,
) -> List[Dict[str, Any]]:
    shaped: List[Dict[str, Any]] = []
    for x in linear_ranked:
        shaped.append(
            {
                "creator_id": x["creator_id"],
                "final_score": float(x["linear_score"]),  # base
                "bm25_score": float(x.get("bm25_score", 0.0)),
                "_source": x["_source"],
            }
        )

    cfg = CrossEncoderConfig(
        model_name=str(ce_model),
        batch_size=int(ce_batch_size),
        max_length=int(ce_max_length),
        device=str(ce_device),
        num_threads=int(ce_num_threads),
        log_every_batches=25,
    )
    ce = get_cross_encoder(cfg)

    return rerank_with_cross_encoder(
        query=str(query),
        hits=shaped,
        top_m=int(ce_rerank_k),
        ce=ce,
        alpha=float(ce_alpha),
    )


def _pick_label_pool(
    rnd: random.Random,
    teacher_ranked: List[Dict[str, Any]],
    linear_ranked: List[Dict[str, Any]],
    label_pool_k: int,
    hard_neg_k: int,
) -> List[str]:
    label_pool_k = max(int(label_pool_k), 0)
    hard_neg_k = max(int(hard_neg_k), 0)

    teacher_ids = [h["creator_id"] for h in teacher_ranked]
    linear_ids = [h["creator_id"] for h in linear_ranked]

    top_teacher = teacher_ids[:label_pool_k] if label_pool_k else []
    top_teacher_set = set(top_teacher)

    # Hard negatives = high-linear but NOT selected by teacher
    window = max(label_pool_k, hard_neg_k, 1)
    candidates = [cid for cid in linear_ids[:window] if cid not in top_teacher_set]
    rnd.shuffle(candidates)
    hard_negs = candidates[:hard_neg_k] if hard_neg_k else []

    pool = list(dict.fromkeys(top_teacher + hard_negs))
    return pool[: (label_pool_k + hard_neg_k)]


def to_graded_relevance(scores: List[float]) -> List[float]:
    if not scores:
        return []
    s_sorted = sorted(scores)
    q50 = s_sorted[int(0.50 * (len(s_sorted) - 1))]
    q75 = s_sorted[int(0.75 * (len(s_sorted) - 1))]
    q90 = s_sorted[int(0.90 * (len(s_sorted) - 1))]

    rel: List[float] = []
    for s in scores:
        if s <= 0.0:
            rel.append(0.0)
        elif s >= q90:
            rel.append(3.0)
        elif s >= q75:
            rel.append(2.0)
        elif s >= q50:
            rel.append(1.0)
        else:
            rel.append(0.0)
    return rel


def apply_label_noise(
    rnd: random.Random,
    rel_map: Dict[str, float],
    noise_down_p: float,
    noise_up_p: float,
) -> Dict[str, float]:
    nd = max(0.0, float(noise_down_p))
    nu = max(0.0, float(noise_up_p))

    out: Dict[str, float] = {}
    for cid, r in rel_map.items():
        rr = float(r)
        if rr >= 2.0 and nd > 0 and rnd.random() < nd:
            rr = max(0.0, rr - 1.0)
        elif rr <= 0.0 and nu > 0 and rnd.random() < nu:
            rr = 1.0

        rr = 0.0 if rr < 0.5 else 1.0 if rr < 1.5 else 2.0 if rr < 2.5 else 3.0
        out[cid] = rr
    return out


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--in-eval", type=Path, default=Path("evalset.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("evalset_labeled.jsonl"))

    ap.add_argument("--index", type=str, default=INDEX_DEFAULT)
    ap.add_argument("--opensearch-url", type=str, default=OPENSEARCH_URL_DEFAULT)
    ap.add_argument("--opensearch-user", type=str, default=OPENSEARCH_USER_DEFAULT)
    ap.add_argument("--opensearch-pass", type=str, default=OPENSEARCH_PASS_DEFAULT)

    ap.add_argument("--candidate-k", type=int, default=2000)
    ap.add_argument("--gold-k", type=int, default=10)

    ap.add_argument("--label-pool-k", type=int, default=300)
    ap.add_argument("--hard-neg-k", type=int, default=200)

    ap.add_argument("--noise-down-p", type=float, default=0.02)
    ap.add_argument("--noise-up-p", type=float, default=0.01)

    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument(
        "--exclude-id-prefix",
        action="append",
        default=[],
        help="Repeatable. Exclude IDs starting with these prefixes from gold.",
    )

    ap.add_argument("--reranker", default=None, help="Optional path to reranker_weights.json")

    # Teacher CE options
    ap.add_argument("--teacher-ce-model", default=None)
    ap.add_argument("--teacher-ce-rerank-k", type=int, default=200)
    ap.add_argument("--teacher-ce-batch-size", type=int, default=32)
    ap.add_argument("--teacher-ce-max-length", type=int, default=192)
    ap.add_argument("--teacher-ce-alpha", type=float, default=1.0)
    ap.add_argument("--teacher-ce-device", type=str, default="auto")
    ap.add_argument("--teacher-ce-num-threads", type=int, default=2)

    args = ap.parse_args()

    exclude_prefixes = args.exclude_id_prefix or []
    if not exclude_prefixes:
        exclude_prefixes = EXCLUDE_ID_PREFIXES_DEFAULT

    rnd = random.Random(int(args.seed))

    eval_items = load_jsonl(args.in_eval)
    client = make_client(args.opensearch_url, args.opensearch_user, args.opensearch_pass)
    reranker = load_reranker(args.reranker)

    # Preload teacher CE once
    teacher_ce_model = args.teacher_ce_model
    if teacher_ce_model:
        _ = get_cross_encoder(
            CrossEncoderConfig(
                model_name=str(teacher_ce_model),
                batch_size=int(args.teacher_ce_batch_size),
                max_length=int(args.teacher_ce_max_length),
                device=str(args.teacher_ce_device),
                num_threads=int(args.teacher_ce_num_threads),
                log_every_batches=25,
            )
        )

    out_lines = 0
    with args.out.open("w", encoding="utf-8") as f_out:
        for ex in eval_items:
            req = ex.get("request") or {}
            desc = str(req.get("description") or "").strip()
            if not desc:
                continue

            constraints = parse_constraints(req)

            body = build_os_query_runtime(desc, constraints)
            res = client.search(index=args.index, body=body, size=int(args.candidate_k))
            hits = (res.get("hits", {}) or {}).get("hits", []) or []
            if not hits:
                continue

            request_meta = {"languages": constraints.languages, "country": constraints.country}

            linear_ranked = linear_score_hits(hits, request_meta=request_meta, reranker=reranker)
            if not linear_ranked:
                continue

            # Teacher ranked
            if teacher_ce_model:
                teacher_ranked = teacher_rerank_with_ce(
                    query=desc,
                    linear_ranked=linear_ranked,
                    ce_model=str(teacher_ce_model),
                    ce_rerank_k=int(args.teacher_ce_rerank_k),
                    ce_batch_size=int(args.teacher_ce_batch_size),
                    ce_max_length=int(args.teacher_ce_max_length),
                    ce_alpha=float(args.teacher_ce_alpha),
                    ce_device=str(args.teacher_ce_device),
                    ce_num_threads=int(args.teacher_ce_num_threads),
                )
                teacher_score = {h["creator_id"]: float(h.get("final_score", 0.0)) for h in teacher_ranked}
            else:
                teacher_ranked = [
                    {"creator_id": x["creator_id"], "final_score": float(x["linear_score"])} for x in linear_ranked
                ]
                teacher_score = {x["creator_id"]: float(x["linear_score"]) for x in linear_ranked}

            teacher_ids = [h["creator_id"] for h in teacher_ranked]

            # GOLD = top teacher ids excluding fixtures
            gold_ids: List[str] = []
            for cid in teacher_ids:
                if is_excluded_id(cid, exclude_prefixes):
                    continue
                gold_ids.append(cid)
                if len(gold_ids) >= int(args.gold_k):
                    break
            if not gold_ids:
                continue

            # Labeled pool
            label_pool_ids = _pick_label_pool(
                rnd=rnd,
                teacher_ranked=[{"creator_id": cid} for cid in teacher_ids],
                linear_ranked=[{"creator_id": x["creator_id"]} for x in linear_ranked],
                label_pool_k=int(args.label_pool_k),
                hard_neg_k=int(args.hard_neg_k),
            )
            label_pool_ids = list(dict.fromkeys(label_pool_ids))

            pool_scores = [float(teacher_score.get(cid, 0.0)) for cid in label_pool_ids]
            graded = to_graded_relevance(pool_scores)

            relevance_graded = {cid: float(graded[i]) for i, cid in enumerate(label_pool_ids)}
            relevance_graded = apply_label_noise(
                rnd=rnd,
                rel_map=relevance_graded,
                noise_down_p=float(args.noise_down_p),
                noise_up_p=float(args.noise_up_p),
            )

            relevance_10 = {cid: float(int(args.gold_k) - i) for i, cid in enumerate(gold_ids)}

            out_obj = {
                "request": req,
                "gold": gold_ids,
                "relevance": relevance_10,
                "relevance_graded": relevance_graded,
                "meta": {
                    "index": args.index,
                    "candidate_k": int(args.candidate_k),
                    "gold_k": int(args.gold_k),
                    "label_pool_k": int(args.label_pool_k),
                    "hard_neg_k": int(args.hard_neg_k),
                    "noise_down_p": float(args.noise_down_p),
                    "noise_up_p": float(args.noise_up_p),
                    "seed": int(args.seed),
                    "excluded_prefixes": exclude_prefixes,
                    "reranker": args.reranker,
                    "teacher_ce_model": teacher_ce_model,
                    "teacher_ce_rerank_k": int(args.teacher_ce_rerank_k),
                    "teacher_ce_batch_size": int(args.teacher_ce_batch_size),
                    "teacher_ce_max_length": int(args.teacher_ce_max_length),
                    "teacher_ce_alpha": float(args.teacher_ce_alpha),
                    "teacher_ce_device": str(args.teacher_ce_device),
                    "labeled_pool_size": len(label_pool_ids),
                    "scoring": "runtime_linear + teacher_ce + hard_negs + label_noise",
                },
            }

            f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            out_lines += 1

    print(
        f"Wrote {args.out} with {out_lines} queries "
        f"(teacher={'CE' if teacher_ce_model else 'linear'}, "
        f"hard_negs={int(args.hard_neg_k)}, noise=({args.noise_down_p},{args.noise_up_p}))."
    )


if __name__ == "__main__":
    main()
