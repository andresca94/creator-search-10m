from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from opensearchpy import OpenSearch

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# ----------------------------
# OpenSearch config (defaults)
# ----------------------------
OPENSEARCH_URL_DEFAULT = "https://localhost:9200"
OPENSEARCH_USER_DEFAULT = "admin"
OPENSEARCH_PASS_DEFAULT = "ChangeThis_ToA_StrongPassword_123!"
INDEX_DEFAULT = "creators_v1"

# When your simulator injects fixtures like cr_gold_1/cr_gold_2 and hard cases,
# we usually want to EXCLUDE them from "gold" so evaluation is on real corpus.
EXCLUDE_ID_PREFIXES_DEFAULT = ["cr_gold_", "cr_hard_"]


# ----------------------------
# Basic parsing helpers
# ----------------------------
def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def parse_iso_ts(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def normalize01(vals: List[float]) -> List[float]:
    if not vals:
        return []
    mn = min(vals)
    mx = max(vals)
    if mx == mn:
        return [0.0 for _ in vals]
    return [(v - mn) / (mx - mn) for v in vals]


# ----------------------------
# Heuristic relevance model
# ----------------------------
SAFETY_WEIGHTS = {
    "adult": 0.6,
    "controversy": 0.4,
    "politics": 0.25,
    "drugs": 0.35,
    "violence": 0.35,
}


@dataclass
class ParsedRequest:
    description: str
    min_followers: int
    geo_country: Optional[str]
    languages: List[str]
    preferences: Dict[str, Any]


def parse_request(req: Dict[str, Any]) -> ParsedRequest:
    desc = (req.get("description") or "").strip()
    hc = req.get("hard_constraints", {}) or {}
    prefs = req.get("preferences", {}) or {}

    min_followers = safe_int(hc.get("min_followers"), 0)

    geo = hc.get("geo") or {}
    geo_country = None
    if isinstance(geo, dict):
        geo_country = geo.get("country") or None
        if geo_country:
            geo_country = str(geo_country).upper()

    langs = hc.get("languages") or []
    if not isinstance(langs, list):
        langs = []
    languages = [str(x).lower() for x in langs if x]

    return ParsedRequest(
        description=desc,
        min_followers=min_followers,
        geo_country=geo_country,
        languages=languages,
        preferences=dict(prefs),
    )


def creator_text(profile: Dict[str, Any]) -> str:
    parts: List[str] = []
    parts.extend(profile.get("verticals") or [])
    parts.extend(profile.get("keywords") or [])
    parts.append(profile.get("bio") or "")

    # recent posts: caption/hashtags/transcript
    for post in (profile.get("recent_posts") or []):
        parts.append(str(post.get("caption", "")))
        for h in (post.get("hashtags") or []):
            parts.append(str(h))
        parts.append(str(post.get("transcript", "")))

    return " ".join(parts)


def compute_recency01(profile: Dict[str, Any], now_utc: datetime, half_life_days: float = 14.0) -> float:
    """
    recency_01 in [0,1] using exponential decay on the most recent post age:
      recency = exp(-ln(2) * age_days / half_life_days)
    """
    newest: Optional[datetime] = None
    for post in (profile.get("recent_posts") or []):
        ts = parse_iso_ts(str(post.get("timestamp") or ""))
        if ts is None:
            continue
        if newest is None or ts > newest:
            newest = ts

    if newest is None:
        return 0.0

    age_days = max((now_utc - newest).total_seconds() / 86400.0, 0.0)
    lam = math.log(2.0) / max(half_life_days, 1e-6)
    return float(math.exp(-lam * age_days))


def compute_engagement_raw(profile: Dict[str, Any]) -> float:
    followers = math.log1p(max(safe_int(profile.get("follower_count"), 0), 0))
    er = safe_float(profile.get("engagement_rate_30d"), 0.0)
    views = math.log1p(max(safe_int(profile.get("avg_views_30d"), 0), 0))
    return 0.6 * followers + 0.3 * er + 0.1 * views


def compute_safety_penalty(profile: Dict[str, Any], avoid_labels: List[str]) -> float:
    labels = set([str(x).lower() for x in (profile.get("content_safety_labels") or []) if x])
    avoid = set([str(x).lower() for x in (avoid_labels or []) if x])

    pen = 0.0
    for lab in labels:
        pen += SAFETY_WEIGHTS.get(lab, 0.0)

    for lab in (labels & avoid):
        pen += 0.35

    if bool(profile.get("is_suspected_spam") or False):
        pen += 0.8

    auth = profile.get("authenticity_score")
    if auth is not None:
        a = safe_float(auth, 1.0)
        pen += max(0.0, 0.6 - a)

    return clamp01(pen)


def lexical_match01(query: str, doc: str) -> float:
    qt = set(tokenize(query))
    dt = set(tokenize(doc))
    if not qt or not dt:
        return 0.0
    inter = len(qt & dt)
    union = len(qt | dt)
    j = inter / union if union else 0.0
    return clamp01(j * 1.5)


def geo_lang_match(profile: Dict[str, Any], pr: ParsedRequest) -> Tuple[float, float]:
    geo_match = 1.0
    if pr.geo_country:
        loc = profile.get("location") or {}
        c = (loc.get("country") or "")
        geo_match = 1.0 if str(c).upper() == pr.geo_country else 0.0

    lang_match = 1.0
    if pr.languages:
        lang = str(profile.get("language") or "").lower()
        lang_match = 1.0 if lang in pr.languages else 0.0

    return geo_match, lang_match


def score_profile(profile: Dict[str, Any], pr: ParsedRequest, now_utc: datetime, eng01: float) -> Tuple[float, Dict[str, Any]]:
    text = lexical_match01(pr.description, creator_text(profile))
    rec = compute_recency01(profile, now_utc)
    geo_m, lang_m = geo_lang_match(profile, pr)

    soft_geo = bool(pr.preferences.get("soft_geo_boost", True))
    soft_lang = bool(pr.preferences.get("soft_language_boost", True))
    avoid_labels = pr.preferences.get("avoid_safety_labels") or []

    safety_pen = compute_safety_penalty(profile, avoid_labels)

    geo_term = 0.05 * geo_m if soft_geo else 0.0
    lang_term = 0.05 * lang_m if soft_lang else 0.0

    base = 0.55 * text + 0.25 * eng01 + 0.15 * rec + geo_term + lang_term
    score = base * (1.0 - 0.7 * safety_pen)

    # min_followers treated as hard (score -> 0)
    if pr.min_followers and safe_int(profile.get("follower_count"), 0) < pr.min_followers:
        score = 0.0

    explain = {
        "features": {
            "text_01": round(text, 6),
            "eng_01": round(eng01, 6),
            "recency_01": round(rec, 6),
            "geo_match": geo_m,
            "lang_match": lang_m,
            "safety_penalty": round(safety_pen, 6),
        },
        "raw_base": round(base, 6),
        "score": round(score, 6),
    }
    return score, explain


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


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# ----------------------------
# OpenSearch retrieval
# ----------------------------
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


def build_os_query(req: Dict[str, Any], candidate_k: int) -> Dict[str, Any]:
    """
    Build an OpenSearch query from the request.
    - min_followers = hard filter (range)
    - description = must multi_match (bio/recent_text)
    - geo/lang are NOT hard-filtered here (so we can still penalize instead of filtering),
      but we can add mild should boosts.
    """
    pr = parse_request(req)

    must: List[Dict[str, Any]] = []
    filt: List[Dict[str, Any]] = []
    should: List[Dict[str, Any]] = []

    desc = pr.description
    if desc:
        must.append(
            {
                "multi_match": {
                    "query": desc,
                    "fields": ["bio", "recent_text"],
                }
            }
        )
    else:
        must.append({"match_all": {}})

    if pr.min_followers:
        filt.append({"range": {"follower_count": {"gte": int(pr.min_followers)}}})

    # Soft boosts (NOT filters)
    if pr.geo_country:
        should.append({"term": {"location.country": {"value": pr.geo_country, "boost": 1.2}}})
    if pr.languages:
        should.append({"terms": {"language": pr.languages}})

    q = {
        "size": candidate_k,
        "track_total_hits": False,
        "_source": {
            "includes": [
                "creator_id",
                "name",
                "language",
                "location.country",
                "location.city",
                "verticals",
                "keywords",
                "bio",
                "recent_text",
                "follower_count",
                "engagement_rate_30d",
                "avg_views_30d",
                "content_safety_labels",
                "is_suspected_spam",
                "authenticity_score",
                "recent_posts",
            ]
        },
        "query": {
            "bool": {
                "must": must,
                "filter": filt,
                "should": should,
                "minimum_should_match": 0,
            }
        },
    }
    return q


def is_excluded_id(cid: str, exclude_prefixes: List[str]) -> bool:
    c = cid or ""
    for p in exclude_prefixes:
        if c.startswith(p):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-eval", type=Path, default=Path("evalset.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("evalset_labeled.jsonl"))

    ap.add_argument("--index", type=str, default=INDEX_DEFAULT)
    ap.add_argument("--opensearch-url", type=str, default=OPENSEARCH_URL_DEFAULT)
    ap.add_argument("--opensearch-user", type=str, default=OPENSEARCH_USER_DEFAULT)
    ap.add_argument("--opensearch-pass", type=str, default=OPENSEARCH_PASS_DEFAULT)

    ap.add_argument("--candidate-k", type=int, default=2000)
    ap.add_argument("--gold-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument(
        "--exclude-id-prefix",
        action="append",
        default=[],
        help="Repeatable. Exclude IDs starting with these prefixes from gold (e.g., cr_gold_, cr_hard_).",
    )
    args = ap.parse_args()

    exclude_prefixes = args.exclude_id_prefix or []
    if not exclude_prefixes:
        exclude_prefixes = EXCLUDE_ID_PREFIXES_DEFAULT

    rnd = random.Random(args.seed)
    now_utc = datetime.now(timezone.utc)

    eval_items = load_jsonl(args.in_eval)
    client = make_client(args.opensearch_url, args.opensearch_user, args.opensearch_pass)

    out_lines = 0
    with args.out.open("w", encoding="utf-8") as f_out:
        for ex in eval_items:
            req = ex.get("request") or {}
            pr = parse_request(req)

            body = build_os_query(req, candidate_k=args.candidate_k)
            res = client.search(index=args.index, body=body)
            hits = (res.get("hits", {}) or {}).get("hits", []) or []

            # Collect candidate profiles from OpenSearch hits
            profiles: List[Dict[str, Any]] = []
            for h in hits:
                src = h.get("_source") or {}
                cid = str(src.get("creator_id") or "")
                if not cid:
                    continue
                profiles.append(src)

            if not profiles:
                # no candidates -> skip
                continue

            # Normalize engagement within candidate pool (query-local)
            eng_raw = [compute_engagement_raw(p) for p in profiles]
            eng01 = normalize01(eng_raw)

            # Score each candidate with your heuristic
            scored: List[Tuple[float, str, float]] = []  # (score, cid, graded_rel later)
            scores_only: List[float] = []
            ids_only: List[str] = []

            for i, p in enumerate(profiles):
                cid = str(p.get("creator_id") or "")
                s, _ = score_profile(p, pr, now_utc, eng01[i])
                scores_only.append(s)
                ids_only.append(cid)

            graded = to_graded_relevance(scores_only)

            # Sort by score desc, then shuffle ties a bit for stability
            order = list(range(len(scores_only)))
            rnd.shuffle(order)
            order.sort(key=lambda j: scores_only[j], reverse=True)

            # Build gold from top scoring REAL corpus ids only (exclude fixtures)
            gold_ids: List[str] = []
            top_indices: List[int] = []
            for j in order:
                cid = ids_only[j]
                if is_excluded_id(cid, exclude_prefixes):
                    continue
                gold_ids.append(cid)
                top_indices.append(j)
                if len(gold_ids) >= args.gold_k:
                    break

            if not gold_ids:
                continue

            # Relevance maps for gold only
            # Relevance maps for gold only
            # Relevance maps
            # - relevance: keep the old 10..1 scale for GOLD only
            relevance_10 = {ids_only[j]: float(args.gold_k - r) for r, j in enumerate(top_indices)}

            # - relevance_graded: label ALL candidates in the pool (so NDCG has real signal)
            #   We already computed: graded = to_graded_relevance(scores_only)
            relevance_graded = {ids_only[i]: float(graded[i]) for i in range(len(ids_only))}

            out_obj = {
                "request": req,
                "gold": gold_ids,
                "relevance": relevance_10,
                "relevance_graded": relevance_graded,
                "meta": {
                    "gold_k": args.gold_k,
                    "candidate_k": args.candidate_k,
                    "seed": args.seed,
                    "index": args.index,
                    "excluded_prefixes": exclude_prefixes,
                },
            }
            f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            out_lines += 1

    print(
        f"Wrote {args.out} with {out_lines} queries "
        f"(gold from OpenSearch candidates only; fixtures excluded)."
    )


if __name__ == "__main__":
    main()
