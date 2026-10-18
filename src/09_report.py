"""Figures for the README."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGS, REPORTS, connect


def weighted_pr(y, s, w):
    order = np.argsort(-s)
    y, s, w = y[order], s[order], w[order]
    tp = np.cumsum(w * y)
    fp = np.cumsum(w * (1 - y))
    return tp / (tp + fp), tp / tp[-1]


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

    d = pd.read_csv(REPORTS / "oof_predictions.csv")
    fig, ax = plt.subplots(figsize=(5, 4))
    for col, label in [("lgb_p", "LightGBM (5-fold CV)"), ("splink_p", "Splink (Fellegi-Sunter, EM)")]:
        p, r = weighted_pr(d.y.values, d[col].values, d.w.values)
        ax.plot(r, p, label=label)
    ax.set_xlabel("Recall (weighted to candidate population)")
    ax.set_ylabel("Precision")
    ax.set_title("Pair-level precision-recall")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGS / "pr_curve.png")

    s = json.loads((REPORTS / "flag_summary.json").read_text())
    fig, ax = plt.subplots(figsize=(5, 3.2))
    vals = [s["chargeoff_rate_all_first_draw"], s["exact_match_baseline"]["chargeoff_rate"], s["model"]["chargeoff_rate"]]
    labels = ["All first-draw\nloans", "Exact-match\nduplicates", "Model-flagged\nduplicates"]
    ax.bar(labels, [v * 100 for v in vals], color=["#9aa5b1", "#5b8db8", "#1f4e79"])
    for i, v in enumerate(vals):
        ax.text(i, v * 100 + 0.3, f"{v * 100:.1f}%", ha="center")
    ax.set_ylabel("Charge-off rate (%)")
    ax.set_title("Flagged loans charge off at ~2x the base rate")
    fig.tight_layout()
    fig.savefig(FIGS / "chargeoff_rate.png")

    con = connect(read_only=True)
    yr = con.execute("""
        WITH c AS (SELECT c.cluster, b.approved FROM ml_clusters c JOIN borrowers b USING (loan_id)),
        k AS (SELECT cluster, min(year(approved)) y0, max(year(approved)) y1 FROM c GROUP BY 1 HAVING count(*) >= 2)
        SELECT y0 || ' + ' || y1 AS years, count(*) n FROM k GROUP BY 1 ORDER BY 2 DESC
    """).df()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(yr.years, yr.n, color="#1f4e79")
    ax.set_xlabel("Flagged businesses")
    ax.set_title("When the duplicate first-draw loans were approved")
    fig.tight_layout()
    fig.savefig(FIGS / "flagged_by_year.png")
    print("figures written to", FIGS)


if __name__ == "__main__":
    main()
