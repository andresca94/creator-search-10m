from __future__ import annotations
import json, random, string
from datetime import datetime, timedelta
from typing import Dict, List, Any

def rid(prefix: str, n: int = 8) -> str:
    return prefix + "_" + "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(n))

VERTICALS = ["Beauty","Music","Sports","Tech","Gaming","Fitness","Food","Travel"]
LANGS = ["en","es","pt"]
COUNTRIES = ["US","CO","MX","BR"]
US_CITIES = ["New Orleans","NYC","LA","Miami","Austin","Chicago"]

KEYWORDS = {
  "Music":["jazz","reggae","live","studio","beats","guitar","concert","salsa"],
  "Sports":["football","matchday","training","stadium","team","championship","soccer"],
  "Beauty":["makeup","skincare","tutorial","glow","routine"],
  "Tech":["ai","gadgets","reviews","coding","cloud"],
  "Gaming":["stream","fps","rpg","console","ranked"],
  "Fitness":["workout","strength","cardio","mobility","nutrition"],
  "Food":["recipes","cooking","tasting","restaurant","chef"],
  "Travel":["beach","city","adventure","hotel","itinerary"],
}
SAFETY = ["none","politics","adult","controversy"]

def make_post(words: List[str], days_ago: int) -> Dict[str, Any]:
    ts = (datetime.utcnow() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    caption = " ".join(random.sample(words, k=min(5,len(words)))) + " ✨"
    hashtags = [f"#{w}" for w in random.sample(words, k=min(2,len(words)))]
    return {
      "post_id": rid("p"),
      "timestamp": ts,
      "caption": caption,
      "hashtags": hashtags,
      "likes": random.randint(50, 25000),
      "comments": random.randint(0, 1500),
      "transcript": None if random.random() < 0.7 else " ".join(words[:4]),
    }

def make_creator(i: int) -> Dict[str, Any]:
    verticals = random.sample(VERTICALS, k=random.randint(1,3))
    if random.random() < 0.30:
        for v in ["Sports","Music"]:
            if v not in verticals: verticals.append(v)
    verticals = list(dict.fromkeys(verticals))

    kw = []
    for v in verticals:
        kw.extend(random.sample(KEYWORDS[v], k=2))
    if random.random() < 0.25:
        kw += ["viral","trending"]

    bio = f"Creator {i}. " + " ".join(random.sample(kw, k=min(8,len(kw))))

    # Mix recency: more recent content tends to matter
    posts = [make_post(kw, days_ago=random.randint(0, 60)) for _ in range(random.randint(1,5))]

    follower_count = int(10 ** random.uniform(4.8, 7.2))  # ~63k..15M
    engagement_rate_30d = round(random.uniform(0.005, 0.15), 4) if random.random() < 0.95 else None
    avg_views_30d = int(follower_count * random.uniform(0.01, 0.6)) if random.random() < 0.9 else None

    safety = "none" if random.random() < 0.92 else random.choice(SAFETY[1:])
    is_spam = random.random() < 0.02
    authenticity_score = round(random.uniform(0.4, 0.99), 3)

    country = random.choice(COUNTRIES)
    city = random.choice(US_CITIES) if country == "US" else None
    language = random.choices(LANGS, weights=[0.7,0.2,0.1])[0]

    return {
      "creator_id": f"cr_{i:07d}",
      "name": rid("name"),
      "verticals": verticals,
      "keywords": kw,
      "language": language,
      "location": {"city": city, "country": country},
      "bio": bio,
      "recent_posts": posts,
      "follower_count": follower_count,
      "engagement_rate_30d": engagement_rate_30d,
      "avg_views_30d": avg_views_30d,
      "content_safety_labels": [safety],
      "is_suspected_spam": is_spam,
      "authenticity_score": authenticity_score,
    }

def inject_hard_cases(creators: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Create explicit 'hard cases' you can cite later:
    1) Keyword stuffing spammer (high text match, spam) -> should be penalized
    2) High engagement but off-topic -> should not dominate
    3) Multilingual mismatch -> should be filtered/boosted
    Returns gold relevant IDs for a few eval queries.
    """
    gold = {}

    # 1) spam keyword stuffing
    spam = make_creator(9_900_001)
    spam["creator_id"] = "cr_hard_spam"
    spam["is_suspected_spam"] = True
    spam["content_safety_labels"] = ["none"]
    spam["bio"] = "football music football music football music " * 10
    spam["keywords"] = ["football","music","football","music"] * 10
    creators.append(spam)

    # 2) high engagement off-topic
    off = make_creator(9_900_002)
    off["creator_id"] = "cr_hard_offtopic"
    off["verticals"] = ["Beauty"]
    off["keywords"] = ["makeup","skincare","tutorial"]
    off["bio"] = "makeup skincare routine glow"
    off["follower_count"] = 8_000_000
    off["engagement_rate_30d"] = 0.14
    off["avg_views_30d"] = 5_000_000
    creators.append(off)

    # 3) multilingual mismatch (es)
    es = make_creator(9_900_003)
    es["creator_id"] = "cr_hard_es"
    es["language"] = "es"
    es["location"] = {"city": None, "country": "US"}
    es["bio"] = "fútbol música partido estadio jazz reggae"
    es["keywords"] = ["fútbol","música","jazz","reggae"]
    creators.append(es)

    # Make some “good” relevant creators
    good = make_creator(9_900_004)
    good["creator_id"] = "cr_gold_1"
    good["language"] = "en"
    good["location"] = {"city": "New Orleans", "country": "US"}
    good["bio"] = "football and music: jazz set after matchday"
    good["keywords"] = ["football","music","jazz","matchday"]
    good["follower_count"] = 1_500_000
    good["engagement_rate_30d"] = 0.06
    creators.append(good)

    good2 = make_creator(9_900_005)
    good2["creator_id"] = "cr_gold_2"
    good2["language"] = "en"
    good2["location"] = {"city": "Austin", "country": "US"}
    good2["bio"] = "studio session then watching the football game"
    good2["keywords"] = ["music","studio","football"]
    good2["follower_count"] = 2_200_000
    creators.append(good2)

    gold["q_football_music_en_us"] = ["cr_gold_1","cr_gold_2"]
    return gold

def write_evalset(gold: Dict[str, List[str]]):
    # Create a few requests with gold & graded relevance
    eval_examples = []

    req1 = {
      "description": "Creators that talk about football and music. Prefer English US creators. Avoid adult/controversy.",
      "hard_constraints": {"min_followers": 1000000, "geo": {"country":"US"}, "languages":["en"]},
      "preferences": {"soft_geo_boost": True, "soft_language_boost": True}
    }
    rel1 = {cid: 3.0 for cid in gold["q_football_music_en_us"]}
    # spam/offtopic should not be relevant
    rel1["cr_hard_spam"] = 0.0
    rel1["cr_hard_offtopic"] = 0.0
    rel1["cr_hard_es"] = 0.0

    eval_examples.append({"request": req1, "gold": gold["q_football_music_en_us"], "relevance": rel1})

    # Another query: allow es, show mismatch handling
    req2 = {
      "description": "football music creators in US",
      "hard_constraints": {"min_followers": 0, "geo": {"country":"US"}, "languages":["en","es"]},
      "preferences": {}
    }
    eval_examples.append({"request": req2, "gold": ["cr_gold_1","cr_gold_2","cr_hard_es"],
                          "relevance": {"cr_gold_1":3,"cr_gold_2":2,"cr_hard_es":2,"cr_hard_spam":0,"cr_hard_offtopic":0}})

    with open("evalset.jsonl","w",encoding="utf-8") as f:
        for ex in eval_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

def main():
    random.seed(7)
    n = 3000
    creators = [make_creator(i) for i in range(n)]
    gold = inject_hard_cases(creators)

    with open("creators.jsonl","w",encoding="utf-8") as f:
        for c in creators:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    request = {
      "description": "Creators that talk about football and music. Prefer English US creators. Avoid adult/controversy.",
      "hard_constraints": {"min_followers": 1000000, "geo": {"country":"US"}, "languages":["en"]},
      "preferences": {"soft_geo_boost": True, "soft_language_boost": True}
    }
    with open("request.json","w",encoding="utf-8") as f:
        json.dump(request, f, ensure_ascii=False, indent=2)

    write_evalset(gold)
    print("Wrote creators.jsonl, request.json, evalset.jsonl")

if __name__ == "__main__":
    main()
