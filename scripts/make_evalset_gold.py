import json
import re
from pathlib import Path
from opensearchpy import OpenSearch

# ---- OpenSearch connection ----
OPENSEARCH_URL = "https://localhost:9200"
INDEX = "creators_v1"
USER = "admin"
PASSWORD = "ChangeThis_ToA_StrongPassword_123!"

# ---- Files ----
INPUT_EVALSET = Path("evalset.jsonl")
OUTPUT_EVALSET = Path("evalset_gold.jsonl")

# ---- Params ----
CANDIDATE_K = 2000   # how many docs we pull from OpenSearch
GOLD_K = 10          # how many become "gold" for the query

# Only accept corpus-like IDs (your simulator uses cr_0000001, etc.)
REAL_ID_RE = re.compile(r"^cr_\d+$")


def build_query(req: dict) -> dict:
    """
    Build an OpenSearch query from request fields.
    Supports:
      - description -> multi_match over bio + recent_text
      - hard_constraints.geo.country -> term filter
      - hard_constraints.languages -> terms filter
      - hard_constraints.min_followers -> range filter
    """
    desc = (req.get("description") or "").strip()

    must = []
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

    filters = []
    hc = req.get("hard_constraints", {}) or {}

    geo = hc.get("geo") or {}
    if isinstance(geo, dict) and geo.get("country"):
        filters.append({"term": {"location.country": geo["country"]}})

    langs = hc.get("languages")
    if isinstance(langs, list) and langs:
        filters.append({"terms": {"language": langs}})

    if hc.get("min_followers") is not None:
        filters.append({"range": {"follower_count": {"gte": int(hc["min_followers"])}}})

    return {"query": {"bool": {"must": must, "filter": filters}}}


def is_real_creator_id(cid: str) -> bool:
    """
    Exclude fixtures such as: cr_gold_1, cr_hard_spam, etc.
    Only allow ids like: cr_0000123 (digits only after cr_)
    """
    return isinstance(cid, str) and REAL_ID_RE.match(cid) is not None


def main() -> None:
    client = OpenSearch(
        OPENSEARCH_URL,
        http_auth=(USER, PASSWORD),
        use_ssl=True,
        verify_certs=False,     # OK for local/demo TLS
        ssl_show_warn=False,
    )

    n_written = 0

    with INPUT_EVALSET.open("r", encoding="utf-8") as f_in, OUTPUT_EVALSET.open(
        "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            if not line.strip():
                continue

            obj = json.loads(line)
            req = obj["request"]

            body = build_query(req)
            res = client.search(index=INDEX, body=body, size=CANDIDATE_K)
            hits = res.get("hits", {}).get("hits", []) or []

            # Build gold strictly from real corpus IDs only
            gold_ids = []
            for h in hits:
                src = h.get("_source") or {}
                cid = src.get("creator_id")

                if is_real_creator_id(cid):
                    gold_ids.append(cid)

                if len(gold_ids) >= GOLD_K:
                    break

            # If we didn't find any real ids, skip this query
            if not gold_ids:
                continue

            # Graded relevance: top=GOLD_K ... bottom=1
            relevance = {cid: float(GOLD_K - i) for i, cid in enumerate(gold_ids)}

            out_obj = {"request": req, "gold": gold_ids, "relevance": relevance}
            f_out.write(json.dumps(out_obj) + "\n")
            n_written += 1

    print(
        f"Wrote {OUTPUT_EVALSET} with {n_written} queries "
        f"(gold ids are REAL corpus ids only; fixtures excluded)"
    )


if __name__ == "__main__":
    main()
