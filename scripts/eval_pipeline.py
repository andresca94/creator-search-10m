import json
import math
import subprocess
import tempfile
from pathlib import Path

EVALSET = Path("evalset_labeled.jsonl")
K = 10
CANDIDATE_K = 2000
INDEX = "creators_v1"

def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))

def run_search(request_obj):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=True) as tf:
        json.dump(request_obj, tf)
        tf.flush()
        cmd = [
            "uv", "run", "creator-search", "search",
            "--index", INDEX,
            "--request", tf.name,
            "--candidate-k", str(CANDIDATE_K),
            "--k", str(K),
        ]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        return json.loads(out)

def main():
    p_sum = r_sum = ndcg_sum = 0.0
    n = 0

    for line in EVALSET.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        ex = json.loads(line)
        req = ex["request"]
        gold = ex["gold"]
        rel = ex.get("relevance_graded") or ex["relevance"]

        res = run_search(req)
        ranked = [x["creator_id"] for x in res["top_k"][:K]]

        gold_set = set(gold)
        hit = sum(1 for cid in ranked if cid in gold_set)

        precision = hit / K
        recall = hit / len(gold_set) if gold_set else 0.0

        gains = [float(rel.get(cid, 0.0)) for cid in ranked]
        ideal = sorted([float(v) for v in rel.values()], reverse=True)[:K]
        denom = dcg(ideal)
        ndcg = (dcg(gains) / denom) if denom > 0 else 0.0

        p_sum += precision
        r_sum += recall
        ndcg_sum += ndcg
        n += 1

    out = {
        "Precision@10": round(p_sum / n, 4) if n else 0.0,
        "Recall@10": round(r_sum / n, 4) if n else 0.0,
        "NDCG@10": round(ndcg_sum / n, 4) if n else 0.0,
        "n_queries": n,
        "k": K,
        "candidate_k": CANDIDATE_K,
        "index": INDEX,
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
