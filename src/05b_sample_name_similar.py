"""Second labeling draw focused on name-similar pairs.

True matches are rare among candidates (most pairs share an office building or a
ZIP code, not an identity), so the first uniform-by-score draw yields few
positives. This draw adds pairs whose best name similarity is >= 0.85, again
uniformly within each score bin. Strata are (score bin, name-similar flag); the
first draw is uniform within these strata too, so the two draws combine into
one stratified sample and weights are recomputed per stratum in 06_train.py.
"""
import pandas as pd

from common import LABELS, connect

TARGET_PER_STRATUM = 130
SEED = 11


def main() -> None:
    con = connect()
    first = pd.read_csv(LABELS / "label_sample.csv", dtype={"a": str, "b": str})
    con.register("first", first[["a", "b"]])
    con.execute("""
        CREATE OR REPLACE TABLE strata AS
        SELECT a, b, least(floor((0.6 * name_jw + 0.4 * addr_jw) * 10), 9)::INT AS h_bin,
               (name_best_jw >= 0.85)::INT AS name_sim
        FROM features
    """)
    have = con.execute("""
        SELECT s.h_bin, s.name_sim, count(*) n FROM first f JOIN strata s USING (a, b) GROUP BY ALL
    """).df()
    extra = []
    for h in range(10):
        got = have.query("h_bin == @h and name_sim == 1").n.sum()
        need = max(TARGET_PER_STRATUM - int(got), 0)
        if need == 0:
            continue
        extra.append(con.execute(f"""
            SELECT s.a, s.b, s.h_bin FROM strata s ANTI JOIN first f USING (a, b)
            WHERE s.h_bin = {h} AND s.name_sim = 1
            ORDER BY hash(s.a || s.b || '{SEED}') LIMIT {need}
        """).df())
    extra = pd.concat(extra)
    con.register("extra", extra)
    df = con.execute("""
        SELECT s.a, s.b, s.h_bin, NULL::DOUBLE AS weight,
               x.raw_name AS a_name, x.raw_address AS a_addr, x.city AS a_city, x.state AS a_state, x.zip5 AS a_zip,
               x.naics AS a_naics, x.business_type AS a_type, x.amount AS a_amt, x.jobs AS a_jobs, x.lender AS a_lender,
               x.approved AS a_date, x.franchise AS a_fr,
               y.raw_name AS b_name, y.raw_address AS b_addr, y.city AS b_city, y.state AS b_state, y.zip5 AS b_zip,
               y.naics AS b_naics, y.business_type AS b_type, y.amount AS b_amt, y.jobs AS b_jobs, y.lender AS b_lender,
               y.approved AS b_date, y.franchise AS b_fr
        FROM extra s JOIN borrowers x ON x.loan_id = s.a JOIN borrowers y ON y.loan_id = s.b
        ORDER BY hash(s.a || s.b)
    """).df()
    df.insert(0, "pair_id", range(int(first.pair_id.max()) + 1, int(first.pair_id.max()) + 1 + len(df)))
    out = pd.concat([first, df], ignore_index=True)
    out.to_csv(LABELS / "label_sample.csv", index=False)
    print(f"added {len(df)} name-similar pairs; total {len(out)}")


if __name__ == "__main__":
    main()
