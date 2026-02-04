import json, math, subprocess, tempfile
from pathlib import Path
from collections import defaultdict

EVALSET = Path("evalset_labeled.jsonl")
K=10
CANDIDATE_K=2000
INDEX="creators_v1"

def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))

def run_search(request_obj):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=True) as tf:
        json.dump(request_obj, tf)
        tf.flush()
        cmd = ["uv","run","creator-search","search",
               "--index",INDEX,"--request",tf.name,
               "--candidate-k",str(CANDIDATE_K),"--k",str(K)]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        return json.loads(out)

rows=[]
for line in EVALSET.read_text(encoding="utf-8").splitlines():
    if not line.strip(): 
        continue
    ex=json.loads(line)
    req=ex["request"]
    gold=set(ex["gold"])
    rel=ex.get("relevance_graded") or ex["relevance"]

    res=run_search(req)
    ranked=[x["creator_id"] for x in res["top_k"][:K]]

    hit=[cid for cid in ranked if cid in gold]
    miss=[cid for cid in list(gold) if cid not in set(ranked)]

    gains=[float(rel.get(cid,0.0)) for cid in ranked]
    ideal=sorted([float(v) for v in rel.values()], reverse=True)[:K]
    denom=dcg(ideal)
    ndcg=(dcg(gains)/denom) if denom>0 else 0.0

    rows.append({
        "q": req.get("description","")[:120],
        "hits": len(hit),
        "ndcg": ndcg,
        "hit_ids": hit,
        "miss_gold": miss[:10],
        "top_ids": ranked[:10],
    })

rows.sort(key=lambda r: (r["hits"], r["ndcg"]))
print("WORST 10 by hits then ndcg:")
for r in rows[:10]:
    print("\n---")
    print("q:", r["q"])
    print("hits:", r["hits"], "ndcg:", round(r["ndcg"],4))
    print("hit_ids:", r["hit_ids"])
    print("miss_gold:", r["miss_gold"])
    print("top_ids:", r["top_ids"])
