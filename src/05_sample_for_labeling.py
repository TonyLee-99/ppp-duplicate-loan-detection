"""Draw a score-stratified sample of candidate pairs for labeling.

There is no ground truth in the PPP data, so a labeled set is built by reviewing
record pairs. Pairs are sampled evenly across bins of a simple similarity score,
so rare hard cases (mid-range scores) are well represented. Each pair keeps a
sampling weight (bin size / sampled count) so metrics can be re-weighted back to
the full candidate population.
"""
from common import LABELS, connect

PER_BIN = 150
SEED = 7


def main() -> None:
    con = connect()
    con.execute("""
        CREATE OR REPLACE TABLE feat_binned AS
        SELECT *, least(floor((0.6 * name_jw + 0.4 * addr_jw) * 10), 9)::INT AS h_bin FROM features
    """)
    print(con.execute("SELECT h_bin, count(*) FROM feat_binned GROUP BY 1 ORDER BY 1").fetchall())
    con.execute(f"""
        CREATE OR REPLACE TABLE label_sample AS
        WITH ranked AS (
            SELECT a, b, h_bin, row_number() OVER (PARTITION BY h_bin ORDER BY hash(a || b || '{SEED}')) AS r,
                   count(*) OVER (PARTITION BY h_bin) AS bin_n
            FROM feat_binned
        )
        SELECT a, b, h_bin, bin_n / least(bin_n, {PER_BIN}) AS weight
        FROM ranked WHERE r <= {PER_BIN}
    """)
    df = con.execute("""
        SELECT row_number() OVER (ORDER BY hash(s.a || s.b)) AS pair_id, s.a, s.b, s.h_bin, s.weight,
               x.raw_name AS a_name, x.raw_address AS a_addr, x.city AS a_city, x.state AS a_state, x.zip5 AS a_zip,
               x.naics AS a_naics, x.business_type AS a_type, x.amount AS a_amt, x.jobs AS a_jobs, x.lender AS a_lender,
               x.approved AS a_date, x.franchise AS a_fr,
               y.raw_name AS b_name, y.raw_address AS b_addr, y.city AS b_city, y.state AS b_state, y.zip5 AS b_zip,
               y.naics AS b_naics, y.business_type AS b_type, y.amount AS b_amt, y.jobs AS b_jobs, y.lender AS b_lender,
               y.approved AS b_date, y.franchise AS b_fr
        FROM label_sample s JOIN borrowers x ON x.loan_id = s.a JOIN borrowers y ON y.loan_id = s.b
        ORDER BY pair_id
    """).df()
    LABELS.mkdir(exist_ok=True)
    df.to_csv(LABELS / "label_sample.csv", index=False)
    print(f"{len(df)} pairs written to labels/label_sample.csv")


if __name__ == "__main__":
    main()
