from __future__ import annotations
import json
from opensearchpy import OpenSearch, helpers
from creator_search.recency import build_recent_text, compute_recency_score

INDEX = "creators_v1"

def main():
    client = OpenSearch("http://localhost:9200")

    def actions():
        with open("creators.jsonl","r",encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                posts = obj.get("recent_posts") or []
                # precompute
                obj["recent_text"] = build_recent_text(posts)
                obj["recency_score"] = compute_recency_score(posts)
                yield {"_index": INDEX, "_id": obj["creator_id"], "_source": obj}

    helpers.bulk(client, actions(), chunk_size=1000, request_timeout=120)
    client.indices.refresh(INDEX)
    print("Ingest complete:", INDEX)

if __name__ == "__main__":
    main()
