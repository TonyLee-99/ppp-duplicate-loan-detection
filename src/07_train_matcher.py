"""Train and evaluate the LightGBM matcher; compare with the Splink baseline.

Evaluation design
-----------------
* Labels come from three stratified draws (see 05*). Every candidate pair falls
  in one stratum = (score bin, name-similar flag, likely-match flag); each
  labeled pair is weighted by N_stratum / n_labeled_stratum so metrics estimate
  performance on the full 12.4M-pair candidate population.
* Pairs labeled "U" (could not tell from public fields) are excluded.
* 5-fold cross-validation grouped by connected components of labeled pairs, so a
  loan never appears in both training and test folds.
* 95% intervals: bootstrap over groups (1,000 resamples).
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from common import LABELS, REPORTS, connect

FEATURES = [
    "name_jw", "name_lev", "name_tok_jacc", "name_best_jw", "name_all_tok_jacc", "name_eq", "dba_cross",
    "addr_jw", "addr_eq", "house_eq", "street_jw", "zip_eq", "city_eq",
    "naics6_eq", "naics3_eq", "btype_eq", "n_individual", "any_franchise", "franchise_eq",
    "same_lender", "amount_ratio", "jobs_ratio", "days_apart", "name_len", "log_name_freq", "log_addr_freq",
]
STRATUM_SQL = """
    least(floor((0.6 * name_jw + 0.4 * addr_jw) * 10), 9)::INT AS h_bin,
    (name_best_jw >= 0.85)::INT AS name_sim,
    (name_best_jw >= 0.9 AND (zip_eq = 1 OR city_eq = 1) AND log_name_freq < ln(20))::INT AS core
"""
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_child_samples=10,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              n_estimators=300, verbose=-1, random_state=0)


def groups_for(df: pd.DataFrame) -> np.ndarray:
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(df.a, df.b):
        parent[find(a)] = find(b)
    roots = [find(a) for a in df.a]
    return pd.factorize(pd.Series(roots))[0]


def weighted_prf(y, pred, w):
    tp = np.sum(w * (y == 1) * (pred == 1))
    fp = np.sum(w * (y == 0) * (pred == 1))
    fn = np.sum(w * (y == 1) * (pred == 0))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def best_threshold(y, s, w):
    grid = np.unique(np.quantile(s, np.linspace(0, 1, 401)))
    scores = [(weighted_prf(y, (s >= t).astype(int), w)[2], t) for t in grid]
    return max(scores)[1]


def bootstrap_f1(y, s, w, g, t, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(g)
    idx_by_g = {k: np.where(g == k)[0] for k in ug}
    out = []
    for _ in range(n):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_by_g[k] for k in pick])
        out.append(weighted_prf(y[idx], (s[idx] >= t).astype(int), w[idx])[2])
    return np.percentile(out, [2.5, 97.5]).tolist()


def main() -> None:
    con = connect()
    sample = pd.read_csv(LABELS / "label_sample.csv", dtype={"a": str, "b": str})[["pair_id", "a", "b"]]
    lab = pd.read_csv(LABELS / "labels_raw.txt", sep=" ", names=["pair_id", "label"], dtype={"label": str})
    lab = sample.merge(lab, on="pair_id")
    lab.to_csv(LABELS / "labels.csv", index=False)
    con.register("lab", lab)
    df = con.execute(f"""
        SELECT l.pair_id, l.label, f.*, {STRATUM_SQL},
               s.match_probability AS splink_p
        FROM lab l JOIN features f USING (a, b)
        LEFT JOIN splink_scores s USING (a, b)
    """).df()
    pop = con.execute(f"SELECT h_bin, name_sim, core, count(*) AS N FROM (SELECT {STRATUM_SQL} FROM features) GROUP BY ALL").df()
    con.close()

    df = df[df.label != "U"].copy()
    df["y"] = (df.label == "1").astype(int)
    df = df.merge(pop, on=["h_bin", "name_sim", "core"])
    df["n"] = df.groupby(["h_bin", "name_sim", "core"]).pair_id.transform("count")
    df["w"] = df.N / df.n
    df["splink_p"] = df.splink_p.fillna(0.0)
    g = groups_for(df)
    y, w = df.y.values, df.w.values

    oof = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=5).split(df, y, g):
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(df.iloc[tr][FEATURES], y[tr])
        oof[te] = m.predict_proba(df.iloc[te][FEATURES])[:, 1]
    df["lgb_p"] = oof

    res = {"n_labeled": int(len(df)), "n_pos": int(y.sum()), "n_uncertain_excluded": int((lab.label == "U").sum()),
           "candidate_pairs": int(pop.N.sum())}
    for name, col in [("lightgbm", "lgb_p"), ("splink", "splink_p")]:
        s = df[col].values
        t = best_threshold(y, s, w)
        p, r, f = weighted_prf(y, (s >= t).astype(int), w)
        p5, r5, f5 = weighted_prf(y, (s >= 0.5).astype(int), w)
        res[name] = {"best_threshold": float(t), "precision": p, "recall": r, "f1": f,
                     "f1_ci95": bootstrap_f1(y, s, w, g, t),
                     "at_0.5": {"precision": p5, "recall": r5, "f1": f5}}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "matcher_metrics.json").write_text(json.dumps(res, indent=2))
    df[["pair_id", "a", "b", "y", "w", "lgb_p", "splink_p", "h_bin", "name_sim", "core"]].to_csv(REPORTS / "oof_predictions.csv", index=False)
    print(json.dumps(res, indent=2))

    final = lgb.LGBMClassifier(**PARAMS).fit(df[FEATURES], y)
    final.booster_.save_model(str(REPORTS / "lightgbm_matcher.txt"))
    imp = pd.Series(final.booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    imp.to_csv(REPORTS / "feature_importance.csv", header=["gain"])

    con = connect()
    rel = con.execute(f"SELECT a, b, {', '.join(FEATURES)} FROM features").df()
    rel["p"] = final.predict_proba(rel[FEATURES])[:, 1]
    con.register("scored", rel[["a", "b", "p"]])
    con.execute("CREATE OR REPLACE TABLE ml_scores AS SELECT * FROM scored")
    print("scored all candidate pairs:", len(rel))


if __name__ == "__main__":
    main()
