import argparse
import json
import math
import random
import string
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-creators", type=int, default=3000, help="How many creators to generate (including hard cases).")
    ap.add_argument("--n-requests", type=int, default=200, help="How many eval requests to generate.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-creators", type=str, default="creators.jsonl")
    ap.add_argument("--out-request", type=str, default="request.json")
    ap.add_argument("--out-evalset", type=str, default="evalset.jsonl")
    return ap.parse_args()


# -----------------------------
# Helpers
# -----------------------------

NOW = datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def pick_weighted(items: List[Tuple[str, float]]) -> str:
    r = random.random() * sum(w for _, w in items)
    acc = 0.0
    for v, w in items:
        acc += w
        if r <= acc:
            return v
    return items[-1][0]


def flatten_recent_text(posts: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for p in posts:
        parts.append(str(p.get("caption", "")))
        for h in p.get("hashtags", []) or []:
            parts.append(str(h))
        parts.append(str(p.get("transcript", "")))
    return " ".join(x for x in parts if x).strip()


def recency_score_from_posts(posts: List[Dict[str, Any]], half_life_days: float = 14.0) -> float:
    """
    Simple recency score in [0,1] based on exponential decay over post age.
    """
    if not posts:
        return 0.0
    hl = max(1e-6, half_life_days)
    weights = []
    for p in posts:
        ts = p.get("timestamp")
        if not ts:
            continue
        try:
            # ts expected like 2025-12-20T00:00:00Z
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            continue
        age_days = max(0.0, (NOW - dt).total_seconds() / 86400.0)
        w = math.exp(-math.log(2) * (age_days / hl))
        weights.append(w)
    if not weights:
        return 0.0
    # map roughly to 0..1
    return clamp01(sum(weights) / len(weights))


def engagement_features(follower_count: int, engagement_rate_30d: float, avg_views_30d: int) -> float:
    """
    Raw engagement used for normalization downstream.
    Keeps numbers reasonable.
    """
    f = math.log1p(max(follower_count, 0))
    v = math.log1p(max(avg_views_30d, 0))
    er = max(0.0, engagement_rate_30d)
    return 0.55 * f + 0.30 * er + 0.15 * v


def random_name() -> str:
    first = random.choice(["Alex", "Jordan", "Taylor", "Sam", "Casey", "Riley", "Jamie", "Morgan", "Avery", "Drew"])
    last = random.choice(["Lopez", "Smith", "Garcia", "Johnson", "Martinez", "Brown", "Davis", "Miller", "Wilson"])
    return f"{first} {last}"


def make_creator_id(i: int) -> str:
    return f"cr_{i:07d}"


def rand_hashtags(keywords: List[str], k: int) -> List[str]:
    kw = random.sample(keywords, k=min(k, len(keywords))) if keywords else []
    return [f"#{x.replace(' ', '')}" for x in kw]


# -----------------------------
# Content dictionaries
# -----------------------------

VERTICALS = [
    "Beauty", "Music", "Sports", "Tech", "Fitness", "Gaming", "Food", "Fashion", "Travel", "Comedy", "Lifestyle"
]

KEYWORDS_BY_VERTICAL = {
    "Music": ["jazz", "music", "live shows", "studio session", "gig", "reggae", "hip hop", "guitar", "beats"],
    "Sports": ["football", "matchday", "training", "team", "highlights", "soccer", "stadium", "fitness"],
    "Tech": ["ai", "ml", "gadgets", "coding", "startups", "cloud", "python", "javascript"],
    "Beauty": ["makeup", "skincare", "lipstick", "beauty tips", "tutorial", "glow"],
    "Fitness": ["workout", "gym", "cardio", "strength", "mobility", "nutrition"],
    "Gaming": ["games", "stream", "fps", "rpg", "speedrun", "esports"],
    "Food": ["recipes", "cooking", "arepas", "coffee", "tacos", "bbq", "dessert"],
    "Fashion": ["outfit", "style", "streetwear", "vintage", "sneakers"],
    "Travel": ["beach", "hiking", "flight", "hotel", "guide", "adventure"],
    "Comedy": ["skits", "jokes", "funny", "parody", "improv"],
    "Lifestyle": ["daily life", "vlog", "routine", "family", "wellness"],
}

COUNTRIES = ["US", "CO", "MX", "BR", "AR", "ES", "GB", "CA"]
CITIES_BY_COUNTRY = {
    "US": ["New York", "Los Angeles", "Chicago", "Miami", "Austin", "New Orleans"],
    "CO": ["Bogotá", "Medellín", "Cali", "Cartagena", "Barranquilla"],
    "MX": ["CDMX", "Guadalajara", "Monterrey"],
    "BR": ["São Paulo", "Rio de Janeiro", "Belo Horizonte"],
    "AR": ["Buenos Aires", "Córdoba"],
    "ES": ["Madrid", "Barcelona", "Valencia"],
    "GB": ["London", "Manchester"],
    "CA": ["Toronto", "Vancouver", "Montreal"],
}

LANG_BY_COUNTRY = {
    "US": ["en", "es"],
    "CO": ["es"],
    "MX": ["es"],
    "BR": ["pt"],
    "AR": ["es"],
    "ES": ["es"],
    "GB": ["en"],
    "CA": ["en", "fr"],
}

SAFETY_LABELS = ["none", "adult", "politics", "controversy", "drugs", "violence"]


def sample_verticals_and_keywords() -> Tuple[List[str], List[str]]:
    # 1-3 verticals
    vs = random.sample(VERTICALS, k=random.randint(1, 3))
    kws: List[str] = []
    for v in vs:
        kws.extend(KEYWORDS_BY_VERTICAL.get(v, []))
    # 3-8 keywords
    if kws:
        keywords = random.sample(list(set(kws)), k=min(random.randint(3, 8), len(set(kws))))
    else:
        keywords = []
    return vs, keywords


def make_posts(keywords: List[str], verticals: List[str]) -> List[Dict[str, Any]]:
    n_posts = random.randint(2, 6)
    posts: List[Dict[str, Any]] = []

    # mix recency: some within last 7 days, some older
    for j in range(n_posts):
        if random.random() < 0.65:
            age_days = random.randint(0, 10)
        else:
            age_days = random.randint(11, 120)

        ts = NOW - timedelta(days=age_days, hours=random.randint(0, 23))
        post_kw = random.sample(keywords, k=min(len(keywords), random.randint(1, 3))) if keywords else []
        caption_bits = []
        if "Sports" in verticals and random.random() < 0.35:
            caption_bits.append("matchday")
        if "Music" in verticals and random.random() < 0.35:
            caption_bits.append("studio session")
        caption_bits.extend(post_kw)
        caption = " ".join(caption_bits) if caption_bits else "new post"
        hashtags = rand_hashtags(post_kw, k=min(3, len(post_kw))) if post_kw else []
        likes = int(max(0, random.gauss(4000, 2500)))
        comments = int(max(0, random.gauss(150, 120)))

        posts.append(
            {
                "post_id": f"p{j+1}",
                "timestamp": iso(ts),
                "caption": caption,
                "hashtags": hashtags,
                "likes": likes,
                "comments": comments,
                "transcript": " ".join(post_kw) if random.random() < 0.4 else "",
            }
        )
    return posts


def sample_safety() -> List[str]:
    # mostly "none", sometimes risky labels
    if random.random() < 0.82:
        return ["none"]
    # 1-2 labels
    labels = random.sample([x for x in SAFETY_LABELS if x != "none"], k=random.randint(1, 2))
    return labels


def build_creator(i: int) -> Dict[str, Any]:
    creator_id = make_creator_id(i)
    name = random_name()
    verticals, keywords = sample_verticals_and_keywords()

    country = random.choice(COUNTRIES)
    city = random.choice(CITIES_BY_COUNTRY[country])
    language = random.choice(LANG_BY_COUNTRY[country])

    follower_count = int(max(0, random.gauss(1_200_000, 900_000)))
    engagement_rate_30d = float(clamp01(abs(random.gauss(0.05, 0.03))))
    avg_views_30d = int(max(0, random.gauss(160_000, 110_000)))

    posts = make_posts(keywords, verticals)
    recent_text = flatten_recent_text(posts)
    recency_score = recency_score_from_posts(posts)

    labels = sample_safety()
    # spam probability higher if authenticity low
    authenticity = clamp01(random.random() ** 0.35)  # skew high
    is_spam = (authenticity < 0.25 and random.random() < 0.5) or (random.random() < 0.02)

    bio_bits = []
    bio_bits.extend(verticals[:])
    bio_bits.extend(keywords[:3])
    bio = " ".join(bio_bits) if bio_bits else "creator"

    return {
        "creator_id": creator_id,
        "name": name,
        "language": language,
        "location": {"country": country, "city": city},
        "verticals": verticals,
        "keywords": keywords,
        "bio": bio,
        "recent_text": recent_text,
        "recent_posts": posts,
        "follower_count": follower_count,
        "engagement_rate_30d": engagement_rate_30d,
        "avg_views_30d": avg_views_30d,
        "content_safety_labels": labels,
        "is_suspected_spam": bool(is_spam),
        "authenticity_score": float(round(authenticity, 4)),
        "recency_score": float(round(recency_score, 6)),
        # helpful for reranking experimentation (not required by mapping)
        "engagement_raw": float(round(engagement_features(follower_count, engagement_rate_30d, avg_views_30d), 6)),
    }


def inject_hard_cases(creators: List[Dict[str, Any]]) -> None:
    """
    Add a handful of deterministic "hard cases":
    - off-topic high engagement
    - spammy but keyword-matching
    - Spanish match in US
    - safety risky but relevant
    """
    base = len(creators)

    # 1) Strongly relevant but safety risky
    c = build_creator(base + 1)
    c["location"] = {"country": "US", "city": "New Orleans"}
    c["language"] = "en"
    c["verticals"] = ["Sports", "Music"]
    c["keywords"] = ["football", "music", "jazz", "matchday", "live shows"]
    c["bio"] = "football + music + jazz nights"
    c["recent_posts"] = [
        {
            "post_id": "p1",
            "timestamp": iso(NOW - timedelta(days=1)),
            "caption": "Game night football + jazz set",
            "hashtags": ["#football", "#jazz", "#music"],
            "likes": 9000,
            "comments": 400,
            "transcript": "football jazz music",
        }
    ]
    c["recent_text"] = flatten_recent_text(c["recent_posts"])
    c["content_safety_labels"] = ["controversy"]
    c["recency_score"] = float(round(recency_score_from_posts(c["recent_posts"]), 6))
    creators.append(c)

    # 2) Spammy keyword match
    c2 = build_creator(base + 2)
    c2["location"] = {"country": "US", "city": "Miami"}
    c2["language"] = "en"
    c2["verticals"] = ["Sports", "Music"]
    c2["keywords"] = ["football", "music", "free", "giveaway", "click", "link"]
    c2["bio"] = "football music free giveaway click link"
    c2["is_suspected_spam"] = True
    c2["authenticity_score"] = 0.12
    creators.append(c2)

    # 3) Off-topic but huge engagement
    c3 = build_creator(base + 3)
    c3["location"] = {"country": "US", "city": "New York"}
    c3["language"] = "en"
    c3["verticals"] = ["Beauty"]
    c3["keywords"] = ["makeup", "skincare", "tutorial", "lipstick"]
    c3["bio"] = "makeup skincare tutorial"
    c3["follower_count"] = 6_000_000
    c3["engagement_rate_30d"] = 0.12
    c3["avg_views_30d"] = 2_000_000
    c3["engagement_raw"] = float(round(engagement_features(c3["follower_count"], c3["engagement_rate_30d"], c3["avg_views_30d"]), 6))
    creators.append(c3)

    # 4) Spanish in US (should pass constraints if allowed)
    c4 = build_creator(base + 4)
    c4["location"] = {"country": "US", "city": "Miami"}
    c4["language"] = "es"
    c4["verticals"] = ["Sports", "Music"]
    c4["keywords"] = ["fútbol", "música", "partido", "jazz"]
    c4["bio"] = "fútbol y música, jazz en vivo"
    creators.append(c4)


def build_request_example() -> Dict[str, Any]:
    return {
        "description": "Creators that talk about football and music. Prefer English US creators. Avoid adult/controversy.",
        "hard_constraints": {
            "min_followers": 1_000_000,
            "geo": {"country": "US"},
            "languages": ["en"],
        },
        "preferences": {
            "soft_geo_boost": True,
            "soft_language_boost": True,
            # safety penalties handled in reranker, not as hard filters
            "avoid_safety_labels": ["adult", "controversy"],
        },
    }


def build_eval_requests(n: int) -> List[Dict[str, Any]]:
    """
    Generate a bunch of diverse requests with:
      - geo/language constraints
      - safety prefs
      - different intents
    These are used as *inputs* for make_evalset_gold.py, which will create gold lists.
    """
    reqs: List[Dict[str, Any]] = []

    topics = [
        ("football and music", ["Sports", "Music"], ["football", "music", "jazz", "matchday"]),
        ("tech and ai coding", ["Tech"], ["ai", "ml", "python", "coding"]),
        ("beauty makeup skincare", ["Beauty"], ["makeup", "skincare", "tutorial"]),
        ("fitness workout gym", ["Fitness"], ["workout", "gym", "strength"]),
        ("gaming streaming esports", ["Gaming"], ["games", "stream", "esports"]),
        ("food recipes cooking", ["Food"], ["recipes", "cooking", "dessert"]),
        ("travel beach adventure", ["Travel"], ["beach", "hiking", "adventure"]),
        ("fashion streetwear", ["Fashion"], ["outfit", "style", "streetwear"]),
        ("comedy skits jokes", ["Comedy"], ["skits", "funny", "parody"]),
        ("lifestyle vlog routine", ["Lifestyle"], ["vlog", "routine", "wellness"]),
    ]

    for _ in range(n):
        topic, _, kws = random.choice(topics)
        country = random.choice(COUNTRIES)
        langs = random.sample(LANG_BY_COUNTRY[country], k=min(len(LANG_BY_COUNTRY[country]), random.randint(1, 2)))

        min_followers = random.choice([0, 50_000, 200_000, 1_000_000])

        avoid = []
        if random.random() < 0.5:
            avoid.append("adult")
        if random.random() < 0.35:
            avoid.append("controversy")
        if random.random() < 0.15:
            avoid.append("politics")

        desc = f"{topic} creators in {country}. " + " ".join(kws[: random.randint(2, 4)])

        reqs.append(
            {
                "request": {
                    "description": desc,
                    "hard_constraints": {
                        "min_followers": int(min_followers),
                        "geo": {"country": country},
                        "languages": langs,
                    },
                    "preferences": {
                        "soft_geo_boost": True,
                        "soft_language_boost": True,
                        "avoid_safety_labels": avoid,
                    },
                }
            }
        )

    return reqs


def main():
    args = parse_args()
    random.seed(args.seed)

    out_creators = Path(args.out_creators)
    out_request = Path(args.out_request)
    out_evalset = Path(args.out_evalset)

    # Generate creators
    creators: List[Dict[str, Any]] = []
    for i in range(args.n_creators):
        creators.append(build_creator(i))

    # Add a few hard cases
    inject_hard_cases(creators)

    # Write creators.jsonl
    with out_creators.open("w", encoding="utf-8") as f:
        for c in creators:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Write a single request.json (handy for quick search)
    req = build_request_example()
    out_request.write_text(json.dumps(req, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Write evalset.jsonl (requests only)
    eval_reqs = build_eval_requests(args.n_requests)
    with out_evalset.open("w", encoding="utf-8") as f:
        for obj in eval_reqs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {out_creators}, {out_request}, {out_evalset}")
    print(f"Creators: {len(creators)} (requested {args.n_creators} + hard cases)")
    print(f"Eval requests: {len(eval_reqs)}")


if __name__ == "__main__":
    main()

