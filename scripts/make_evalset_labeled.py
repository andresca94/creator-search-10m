from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from opensearchpy import OpenSearch

# ---------------------------------------------------------------------
# GOLD + labels generated with a *teacher* that matches (or extends) runtime
# Runtime = OpenSearch retrieve + Linear rerank
# Teacher  = Runtime + Cross-Encoder rerank on top-M
#
# Additionally:
# - hard negatives: include “high linear, low teacher” items in the labeled pool
# - label noise: flip a small % of labels up/down for realism
#
# IMPORTANT: Cross-Encoder must be cached/loaded ONCE per run, not per query.
# ---------------------------------------------------------------------

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


# ----------------------------
# IO helpers
# ----------------------------
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


# ----------------------------
# Runtime scoring (linear)
# ----------------------------
def linear_score_hits(
    hits: List[Dict[str, Any]],
    request_meta: Dict[str, Any],
    reranker: LinearReranker,
) -> List[Dict[str, Any]]:
    """
    Score OpenSearch hits using the SAME feature extraction + linear reranker as runtime.
    Returns list sorted by linear score desc.
    Each element includes:
      - creator_id
      - linear_score
      - bm25_score
      - _source  (for CE doc text)
      - _hit     (raw OS hit)
    """
    out: List[Dict[str, Any]] = []
    for h in hits:
        src = h.get("_source") or {}
        cid = str(src.get("creator_id") or "")
        if not cid:
            continue

        # Ensure recency_score exists for features.py (if your index stores it, no-op)
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


# ----------------------------
# Teacher scoring (CE)
# ----------------------------
def teacher_rerank_with_ce(
    query: str,
    linear_ranked: List[Dict[str, Any]],
    ce_model: str,
    ce_rerank_k: int,
    ce_batch_size: int,
    ce_max_length: int,
    ce_alpha: float,
    ce_device: str,
) -> List[Dict[str, Any]]:
    """
    Uses CE on top-M (by linear), combines with linear via alpha, returns sorted list.

    We reuse creator_search.cross_encoder.rerank_with_cross_encoder().
    That helper expects "hits" entries to look like runtime ranked dicts with:
      - final_score (we'll feed linear_score as final_score)
      - _source (doc text)
      - creator_id
    """
    # Build the hits shape expected by rerank_with_cross_encoder
    shaped = []
    for x in linear_ranked:
        shaped.append(
            {
                "creator_id": x["creator_id"],
                "final_score": float(x["linear_score"]),  # base score
                "bm25_score": float(x.get("bm25_score", 0.0)),
                "_source": x["_source"],
            }
        )

    cfg = CrossEncoderConfig(
        model_name=str(ce_model),
        batch_size=int(ce_batch_size),
        max_length=int(ce_max_length),
        device=str(ce_device),
    )
    ce = get_cross_encoder(cfg)  # cached per-process

    reranked = rerank_with_cross_encoder(
        query=str(query),
        hits=shaped,
        top_m=int(ce_rerank_k),
        ce=ce,
        alpha=float(ce_alpha),
    )

    # rerank_with_cross_encoder returns a list sorted by new final_score desc
    return reranked


# ----------------------------
# Labeling pool selection
# ----------------------------
def _pick_label_pool(
    rnd: random.Random,
    teacher_ranked: List[Dict[str, Any]],
    linear_ranked: List[Dict[str, Any]],
    label_pool_k: int,
    hard_neg_k: int,
) -> List[str]:
    """
    label pool = top label_pool_k by teacher score + hard negatives (high linear, low teacher)

    Hard negatives definition:
      candidates that are in top ~label_pool_k of linear but NOT in top ~label_pool_k of teacher
      (and we add up to hard_neg_k of them).
    """
    label_pool_k = max(int(label_pool_k), 0)
    hard_neg_k = max(int(hard_neg_k), 0)

    teacher_ids = [h["creator_id"] for h in teacher_ranked]
    linear_ids = [h["creator_id"] for h in linear_ranked]

    top_teacher = teacher_ids[:label_pool_k] if label_pool_k else []
    top_teacher_set = set(top_teacher)

    # high-linear items not selected by teacher
    candidates = [cid for cid in linear_ids[: max(label_pool_k, hard_neg_k, 1)] if cid not in top_teacher_set]
    rnd.shuffle(candidates)
    hard_negs = candidates[:hard_neg_k] if hard_neg_k else []

    pool = list(dict.fromkeys(top_teacher + hard_negs))  # stable unique
    return pool[: max(label_pool_k, 0) + max(hard_neg_k, 0)]


# ----------------------------
# Graded relevance + noise
# ----------------------------
def to_graded_relevance(scores: List[float]) -> List[float]:
    """
    Convert scores into {0,1,2,3} by query-local quantiles.
    """
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
    """
    More realistic labels:
      - noise_down_p: with this prob, demote strong positives (>=2) by 1 (2->1, 3->2)
      - noise_up_p  : with this prob, promote negatives (0) to weak positive (1)

    Keeps values in {0,1,2,3}.
    """
    nd = max(0.0, float(noise_down_p))
    nu = max(0.0, float(noise_up_p))

    out: Dict[str, float] = {}
    for cid, r in rel_map.items():
        rr = float(r)
        if rr >= 2.0 and nd > 0 and rnd.random() < nd:
            rr = max(0.0, rr - 1.0)
        elif rr <= 0.0 and nu > 0 and rnd.random() < nu:
            rr = 1.0
        # clamp to {0,1,2,3}
        rr = 0.0 if rr < 0.5 else 1.0 if rr < 1.5 else 2.0 if rr < 2.5 else 3.0
        out[cid] = rr
    return out


# ----------------------------
# Main
# ----------------------------
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

    # Labeled pool controls
    ap.add_argument("--label-pool-k", type=int, default=500, help="How many top teacher items to label (per query).")
    ap.add_argument("--hard-neg-k", type=int, default=250, help="Add this many hard negatives to the labeled pool.")
    ap.add_argument("--noise-down-p", type=float, default=0.0, help="Prob of demoting positives (>=2) by 1.")
    ap.add_argument("--noise-up-p", type=float, default=0.0, help="Prob of promoting negatives (0) to 1.")

    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument(
        "--exclude-id-prefix",
        action="append",
        default=[],
        help="Repeatable. Exclude IDs starting with these prefixes from gold (e.g., cr_gold_, cr_hard_).",
    )

    # Linear reranker weights (optional)
    ap.add_argument("--reranker", default=None, help="Optional path to reranker_weights.json")

    # Teacher CE options
    ap.add_argument(
        "--teacher-ce-model",
        default=None,
        help="HF model id for cross-encoder teacher (e.g., cross-encoder/ms-marco-MiniLM-L-12-v2).",
    )
    ap.add_argument("--teacher-ce-rerank-k", type=int, default=200, help="Teacher CE reranks top-M from linear.")
    ap.add_argument("--teacher-ce-batch-size", type=int, default=8)
    ap.add_argument("--teacher-ce-max-length", type=int, default=256)
    ap.add_argument("--teacher-ce-alpha", type=float, default=1.0, help="1.0=teacher uses CE only; 0.0=linear only.")
    ap.add_argument("--teacher-ce-device", type=str, default="cpu", help="cpu (t3.large has no GPU).")

    args = ap.parse_args()

    exclude_prefixes = args.exclude_id_prefix or []
    if not exclude_prefixes:
        exclude_prefixes = EXCLUDE_ID_PREFIXES_DEFAULT

    rnd = random.Random(int(args.seed))

    eval_items = load_jsonl(args.in_eval)
    client = make_client(args.opensearch_url, args.opensearch_user, args.opensearch_pass)

    reranker = load_reranker(args.reranker)

    # Pre-load teacher CE ONCE (cached inside get_cross_encoder too, but we trigger here so
    # you see it loading once and not per query).
    teacher_ce_model = args.teacher_ce_model
    if teacher_ce_model:
        _ = get_cross_encoder(
            CrossEncoderConfig(
                model_name=str(teacher_ce_model),
                batch_size=int(args.teacher_ce_batch_size),
                max_length=int(args.teacher_ce_max_length),
                device=str(args.teacher_ce_device),
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

            # Build query using runtime definition
            body = build_os_query_runtime(desc, constraints)
            res = client.search(index=args.index, body=body, size=int(args.candidate_k))
            hits = (res.get("hits", {}) or {}).get("hits", []) or []
            if not hits:
                continue

            request_meta = {
                "languages": constraints.languages,
                "country": constraints.country,
            }

            # Runtime linear scoring
            linear_ranked_full = linear_score_hits(hits, request_meta=request_meta, reranker=reranker)
            if not linear_ranked_full:
                continue

            # Build teacher ranked list
            if teacher_ce_model:
                teacher_ranked = teacher_rerank_with_ce(
                    query=desc,
                    linear_ranked=linear_ranked_full,
                    ce_model=str(teacher_ce_model),
                    ce_rerank_k=int(args.teacher_ce_rerank_k),
                    ce_batch_size=int(args.teacher_ce_batch_size),
                    ce_max_length=int(args.teacher_ce_max_length),
                    ce_alpha=float(args.teacher_ce_alpha),
                    ce_device=str(args.teacher_ce_device),
                )
                # teacher_ranked entries are dicts w/ creator_id + final_score
                teacher_ids = [h["creator_id"] for h in teacher_ranked]
                teacher_score = {h["creator_id"]: float(h.get("final_score", 0.0)) for h in teacher_ranked}
            else:
                # No CE teacher => teacher == linear
                teacher_ids = [x["creator_id"] for x in linear_ranked_full]
                teacher_score = {x["creator_id"]: float(x["linear_score"]) for x in linear_ranked_full}

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

            # labeled pool = top teacher + hard negatives
            label_pool_ids = _pick_label_pool(
                rnd=rnd,
                teacher_ranked=[{"creator_id": cid} for cid in teacher_ids],
                linear_ranked=[{"creator_id": cid} for cid in [x["creator_id"] for x in linear_ranked_full]],
                label_pool_k=int(args.label_pool_k),
                hard_neg_k=int(args.hard_neg_k),
            )
            # Deduplicate and keep order
            label_pool_ids = list(dict.fromkeys(label_pool_ids))

            # Scores in labeled pool (teacher scores)
            pool_scores: List[float] = [float(teacher_score.get(cid, 0.0)) for cid in label_pool_ids]
            graded = to_graded_relevance(pool_scores)

            relevance_graded = {cid: float(graded[i]) for i, cid in enumerate(label_pool_ids)}
            relevance_graded = apply_label_noise(
                rnd=rnd,
                rel_map=relevance_graded,
                noise_down_p=float(args.noise_down_p),
                noise_up_p=float(args.noise_up_p),
            )

            # Rank-based relevance for GOLD only (gold_k..1)
            gold_k = int(args.gold_k)
            relevance_10 = {cid: float(gold_k - i) for i, cid in enumerate(gold_ids)}

            out_obj = {
                "request": req,
                "gold": gold_ids,
                "relevance": relevance_10,
                "relevance_graded": relevance_graded,  # sparse, realistic labels
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
                    "scoring": "runtime_linear + optional_teacher_cross_encoder + hard_negs + label_noise",
                    "labeled_pool_size": len(label_pool_ids),
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
