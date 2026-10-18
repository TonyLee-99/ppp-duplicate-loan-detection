"""Splink (Fellegi-Sunter) baseline on the same candidate pairs.

The blocking keys from 03_blocking.py are materialised as columns (NULL when the
block is larger than the cap), so Splink generates the same candidate set.
Parameters are estimated without labels: u by random sampling, m by EM.
Output: table splink_scores(a, b, match_weight, match_probability).
"""
import time

import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

from common import DATA, connect

MAX_BLOCK = 50


def main() -> None:
    t0 = time.time()
    con = connect()
    con.execute(f"""
        CREATE OR REPLACE TABLE splink_input AS
        WITH k AS (
            SELECT loan_id AS unique_id, name_std, addr_std, zip5, city, state, naics, business_type, house_no,
                   zip5 || '|' || name_pre6 AS k1,
                   CASE WHEN house_no <> '' THEN zip5 || '|' || house_no || '|' || split_part(street, ' ', 1) END AS k2,
                   CASE WHEN length(name_std) >= 8 THEN state || '|' || name_std END AS k3,
                   CASE WHEN length(name_std) >= 8 THEN state || '|' || array_to_string(list_sort(string_split(name_std, ' ')), ' ') END AS k4
            FROM borrowers
        )
        SELECT unique_id, name_std, addr_std, zip5, city, state, naics, business_type, house_no,
            CASE WHEN count(*) OVER (PARTITION BY k1) <= {MAX_BLOCK} THEN k1 END AS k1,
            CASE WHEN count(*) OVER (PARTITION BY k2) <= {MAX_BLOCK} THEN k2 END AS k2,
            CASE WHEN count(*) OVER (PARTITION BY k3) <= {MAX_BLOCK} THEN k3 END AS k3,
            CASE WHEN count(*) OVER (PARTITION BY k4) <= {MAX_BLOCK} THEN k4 END AS k4
        FROM k
    """)
    parquet = DATA / "splink_input.parquet"
    con.execute(f"COPY splink_input TO '{parquet}' (FORMAT parquet)")
    con.close()

    db = DuckDBAPI()
    db.duckdb_con.execute("SET memory_limit='10GB'")
    db.duckdb_con.execute(f"CREATE TABLE borrowers_in AS SELECT * FROM read_parquet('{parquet}')")
    sdf = db.register("borrowers_in", dataset_display_name="borrowers")
    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=[
            cl.JaroWinklerAtThresholds("name_std", [0.97, 0.9, 0.8]).configure(term_frequency_adjustments=True),
            cl.JaroWinklerAtThresholds("addr_std", [0.97, 0.9, 0.8]),
            cl.ExactMatch("zip5"),
            cl.ExactMatch("city"),
            cl.ExactMatch("naics"),
            cl.ExactMatch("business_type"),
            cl.ExactMatch("house_no"),
        ],
        blocking_rules_to_generate_predictions=[block_on(k) for k in ("k1", "k2", "k3", "k4")],
        retain_matching_columns=False,
    )
    linker = Linker(sdf, settings)
    linker.training.estimate_probability_two_random_records_match(
        [block_on("name_std", "addr_std", "zip5")], recall=0.8
    )
    linker.training.estimate_u_using_random_sampling(max_pairs=2e7)
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("zip5", "house_no"))
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("state", "name_std"))
    pred = linker.inference.predict()
    out_path = DATA / "splink_scores.parquet"
    db.duckdb_con.execute(f"""
        COPY (SELECT least(unique_id_l, unique_id_r) AS a, greatest(unique_id_l, unique_id_r) AS b,
                     match_weight, match_probability FROM {pred.physical_name})
        TO '{out_path}' (FORMAT parquet)
    """)
    con = connect()
    con.execute(f"CREATE OR REPLACE TABLE splink_scores AS SELECT * FROM read_parquet('{out_path}')")
    n, = con.execute("SELECT count(*) FROM splink_scores").fetchone()
    print(f"splink scored {n:,} pairs in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
