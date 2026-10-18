"""Cluster matched loans into businesses and flag possible duplicate first-draw loans.

* Edges: candidate pairs with LightGBM probability >= THRESHOLD (chosen on
  cross-validated predictions for ~85% weighted pair precision).
* Pairs where the earlier loan was already closed (status date) before the later
  loan was approved are dropped: those look like a returned loan followed by a
  re-application, not two loans held at the same time.
* Connected components with 2+ loans = flagged businesses.
* Baseline: exact match on standardized name + address + ZIP.
"""
import json

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from common import REPORTS, connect

THRESHOLD = 0.8


def components(edges: pd.DataFrame) -> pd.DataFrame:
    ids = pd.Index(pd.unique(edges[["a", "b"]].values.ravel()))
    ia, ib = ids.get_indexer(edges.a), ids.get_indexer(edges.b)
    n = len(ids)
    m = coo_matrix((np.ones(len(ia)), (ia, ib)), shape=(n, n))
    _, lab = connected_components(m, directed=False)
    return pd.DataFrame({"loan_id": ids, "cluster": lab})


def summarize(con, table: str) -> dict:
    return con.execute(f"""
        WITH c AS (
            SELECT c.cluster, b.loan_id, b.amount, b.lender, b.approved, b.status,
                   row_number() OVER (PARTITION BY c.cluster ORDER BY b.approved, b.loan_id) AS rk
            FROM {table} c JOIN borrowers b USING (loan_id)
        ), agg AS (
            SELECT cluster, count(*) AS n_loans, count(DISTINCT lender) AS n_lenders,
                   sum(amount) FILTER (WHERE rk > 1) AS extra_amount,
                   sum(amount) AS total_amount,
                   avg((status = 'Charged Off')::INT) AS chargeoff_rate
            FROM c GROUP BY cluster HAVING count(*) >= 2
        )
        SELECT count(*) AS businesses, sum(n_loans) AS loans, sum(n_loans - 1) AS extra_loans,
               sum(extra_amount) AS extra_amount, avg((n_lenders > 1)::INT) AS share_multi_lender,
               sum(chargeoff_rate * n_loans) / sum(n_loans) AS chargeoff_rate
        FROM agg
    """).df().iloc[0].to_dict()


def main() -> None:
    con = connect()
    edges = con.execute(f"""
        SELECT s.a, s.b FROM ml_scores s
        JOIN borrowers x ON x.loan_id = s.a JOIN borrowers y ON y.loan_id = s.b
        WHERE s.p >= {THRESHOLD}
          AND NOT (
              (x.approved < y.approved AND x.status_date IS NOT NULL AND x.status_date < y.approved) OR
              (y.approved < x.approved AND y.status_date IS NOT NULL AND y.status_date < x.approved)
          )
    """).df()
    cl = components(edges)
    con.register("cl_df", cl)
    con.execute("CREATE OR REPLACE TABLE ml_clusters AS SELECT * FROM cl_df")

    con.execute("""
        CREATE OR REPLACE TABLE exact_clusters AS
        SELECT loan_id, dense_rank() OVER (ORDER BY name_std, addr_std, zip5) AS cluster
        FROM borrowers
        WHERE (name_std, addr_std, zip5) IN (
            SELECT (name_std, addr_std, zip5) FROM borrowers WHERE addr_std <> '' GROUP BY ALL HAVING count(*) >= 2
        )
    """)
    ml = summarize(con, "ml_clusters")
    ex = summarize(con, "exact_clusters")
    base_rate, = con.execute("SELECT avg((status = 'Charged Off')::INT) FROM borrowers").fetchone()

    res = {"threshold": THRESHOLD, "edges": int(len(edges)), "model": ml, "exact_match_baseline": ex,
           "chargeoff_rate_all_first_draw": base_rate,
           "ratio_flagged_vs_exact": ml["businesses"] / ex["businesses"]}
    (REPORTS / "flag_summary.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))

    # Random sample of flagged businesses for manual review.
    con.execute("""
        CREATE OR REPLACE TABLE review_sample AS
        WITH k AS (SELECT cluster FROM ml_clusters GROUP BY cluster HAVING count(*) >= 2
                   ORDER BY hash(cluster || 'review') LIMIT 150)
        SELECT c.cluster, b.raw_name, b.raw_address, b.city, b.state, b.zip5, b.naics, b.business_type,
               b.amount, b.jobs, b.lender, b.approved, b.status
        FROM ml_clusters c JOIN k USING (cluster) JOIN borrowers b USING (loan_id)
        ORDER BY c.cluster, b.approved
    """)
    con.execute(f"COPY review_sample TO '{REPORTS / 'review_sample.csv'}' (HEADER)")


if __name__ == "__main__":
    main()
