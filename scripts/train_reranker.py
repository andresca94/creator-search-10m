from __future__ import annotations
import json
import numpy as np
from sklearn.linear_model import LogisticRegression

FEATURES = ["text_01","eng_01","recency_01","lang_match","geo_match","safety_penalty"]

# train_examples.jsonl:
# {"x": {"text_01":..., ...}, "y": 0/1}
def main():
    X, y = [], []
    with open("train_examples.jsonl","r",encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            x = ex["x"]
            X.append([x.get(k,0.0) for k in FEATURES])
            y.append(int(ex["y"]))
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    clf = LogisticRegression(max_iter=300)
    clf.fit(X, y)

    weights = {k: float(w) for k,w in zip(FEATURES, clf.coef_[0])}
    bias = float(clf.intercept_[0])

    with open("reranker_weights.json","w",encoding="utf-8") as f:
        json.dump({"weights": weights, "bias": bias}, f, indent=2)

    print("Saved reranker_weights.json")

if __name__ == "__main__":
    main()
