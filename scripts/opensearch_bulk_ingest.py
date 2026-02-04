from __future__ import annotations
import json
from opensearchpy import OpenSearch, helpers

# If your repo already has these, keep them. Otherwise these imports must exist.
from creator_search.recency import build_recent_text, compute_recency_score

INDEX = "creators_v1"
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

def main():
    client = make_client()

    def actions():
        with open("creators.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                posts = obj.get("recent_posts") or []
                obj["recent_text"] = build_recent_text(posts)
                obj["recency_score"] = compute_recency_score(posts)
                yield {"_index": INDEX, "_id": obj["creator_id"], "_source": obj}

    helpers.bulk(client, actions(), chunk_size=1000, request_timeout=120)
    client.indices.refresh(index=INDEX)
    print("Ingest complete:", INDEX)

if __name__ == "__main__":
    main()
