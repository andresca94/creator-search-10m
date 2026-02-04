from opensearchpy import OpenSearch

INDEX = "creators_v1"

def main():
    client = OpenSearch(hosts=["http://localhost:9200"])

    # NOTE: must use keyword arg `index=...`
    if client.indices.exists(index=INDEX):
        client.indices.delete(index=INDEX)

    body = {
        "settings": {"index": {"number_of_shards": 3, "number_of_replicas": 0}},
        "mappings": {
            "properties": {
                "creator_id": {"type": "keyword"},
                "name": {"type": "keyword"},
                "language": {"type": "keyword"},
                "location": {
                    "properties": {
                        "country": {"type": "keyword"},
                        "city": {"type": "keyword"},
                    }
                },
                "verticals": {"type": "keyword"},
                "keywords": {"type": "keyword"},
                "bio": {"type": "text"},
                "recent_text": {"type": "text"},
                "follower_count": {"type": "long"},
                "engagement_rate_30d": {"type": "float"},
                "avg_views_30d": {"type": "long"},
                "content_safety_labels": {"type": "keyword"},
                "is_suspected_spam": {"type": "boolean"},
                "authenticity_score": {"type": "float"},
                "recency_score": {"type": "float"},
                "recent_posts": {"type": "object", "enabled": False},
            }
        },
    }

    # NOTE: must use keyword arg `index=...`
    client.indices.create(index=INDEX, body=body)
    print("Created index", INDEX)

if __name__ == "__main__":
    main()
