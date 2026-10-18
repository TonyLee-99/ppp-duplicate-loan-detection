"""Third labeling draw: the "likely match" region.

Pairs with a very similar, uncommon name (best name JW >= 0.9, name appears
< 20 times nationally) in the same ZIP or city. This is where most true matches
and the hardest look-alikes (related LLCs, family members) live. The stratum
flag refines the earlier strata, so all three draws stay uniform within the
final (score bin, name-similar, core) strata.
"""
import pandas as pd

from common import LABELS, connect

PER_BIN = 110
SEED = 23


def main() -> None:
    con = connect()
    prev = pd.read_csv(LABELS / "label_sample.csv", dtype={"a": str, "b": str})
    con.register("prev", prev[["a", "b"]])
    extra = con.execute(f"""
        WITH core AS (
            SELECT a, b, least(floor((0.6 * name_jw + 0.4 * addr_jw) * 10), 9)::INT AS h_bin
            FROM features
            WHERE name_best_jw >= 0.9 AND (zip_eq = 1 OR city_eq = 1) AND log_name_freq < ln(20)
        ), ranked AS (
            SELECT c.*, row_number() OVER (PARTITION BY h_bin ORDER BY hash(a || b || '{SEED}')) AS r
            FROM core c ANTI JOIN prev p USING (a, b)
        )
        SELECT a, b, h_bin FROM ranked WHERE r <= {PER_BIN}
    """).df()
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
    start = int(prev.pair_id.max()) + 1
    df.insert(0, "pair_id", range(start, start + len(df)))
    out = pd.concat([prev, df], ignore_index=True)
    out.to_csv(LABELS / "label_sample.csv", index=False)
    print(f"added {len(df)} likely-match pairs; total {len(out)}")


if __name__ == "__main__":
    main()
