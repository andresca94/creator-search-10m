from creator_search.constraints import parse_constraints
from creator_search.safety import safety_penalty
from creator_search.recency import compute_recency_score

def test_constraints_parse():
    req = {"hard_constraints": {"min_followers": 10, "geo": {"country":"US"}, "languages":["en","es"]}}
    c = parse_constraints(req)
    assert c.min_followers == 10
    assert c.country == "US"
    assert c.languages == ["en","es"]

def test_safety_penalty_spam_high():
    assert safety_penalty(["none"], True, 0.9) >= 0.75

def test_recency_score_new_gt_old():
    old = [{"timestamp":"2024-01-01"}]
    new = [{"timestamp":"2026-02-01"}]
    assert compute_recency_score(new) > compute_recency_score(old)
