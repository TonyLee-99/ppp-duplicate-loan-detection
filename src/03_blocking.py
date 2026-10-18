"""Generate candidate pairs with multi-pass blocking.

Comparing every pair of 8.6M first-draw loans would mean ~3.7e13 comparisons.
Each blocking rule only compares loans that share a cheap key; the union of the
rules is the candidate set. Very large blocks (chains, generic names) are capped
so they do not explode the pair count.
"""
from common import connect

MAX_BLOCK = 50

RULES = {
    # same ZIP and same first six letters of the name
    "zip_name6": "zip5 || '|' || name_pre6",
    # same street address (house number + street) in the same ZIP
    "zip_address": "CASE WHEN house_no <> '' THEN zip5 || '|' || house_no || '|' || split_part(street, ' ', 1) END",
    # same standardized name anywhere in the state (catches moves)
    "state_name": "CASE WHEN length(name_std) >= 8 THEN state || '|' || name_std END",
    # same name tokens in any order within the state (SMITH JOHN vs JOHN SMITH)
    "state_name_sorted": "CASE WHEN length(name_std) >= 8 THEN state || '|' || array_to_string(list_sort(string_split(name_std, ' ')), ' ') END",
    # legal name on one loan equals the DBA name on another
    "state_dba": "CASE WHEN length(dba_std) >= 5 THEN state || '|' || dba_std END",
}


def main() -> None:
    con = connect()
    n, = con.execute("SELECT count(*) FROM borrowers").fetchone()
    print(f"all possible pairs: {n * (n - 1) / 2:.3e}")
    con.execute("CREATE OR REPLACE TABLE candidates (a VARCHAR, b VARCHAR, rule VARCHAR)")
    for rule, key in RULES.items():
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE k AS
            SELECT loan_id, {key} AS key FROM borrowers WHERE {key} IS NOT NULL
        """)
        if rule == "state_dba":
            # match the DBA key against legal-name keys in the same state
            con.execute("""
                CREATE OR REPLACE TEMP TABLE k2 AS
                SELECT loan_id, state || '|' || name_std AS key FROM borrowers WHERE length(name_std) >= 5
            """)
            con.execute(f"""
                INSERT INTO candidates
                SELECT least(x.loan_id, y.loan_id), greatest(x.loan_id, y.loan_id), '{rule}'
                FROM k x JOIN k2 y USING (key)
                WHERE x.loan_id <> y.loan_id
                  AND key IN (SELECT key FROM k2 GROUP BY key HAVING count(*) <= {MAX_BLOCK})
            """)
        else:
            con.execute(f"""
                INSERT INTO candidates
                WITH ok AS (SELECT key FROM k GROUP BY key HAVING count(*) BETWEEN 2 AND {MAX_BLOCK})
                SELECT x.loan_id, y.loan_id, '{rule}'
                FROM k x JOIN k y USING (key)
                WHERE key IN (SELECT key FROM ok) AND x.loan_id < y.loan_id
            """)
        c, = con.execute(f"SELECT count(*) FROM candidates WHERE rule = '{rule}'").fetchone()
        print(f"{rule:20s} {c:>12,} pairs")
    con.execute("""
        CREATE OR REPLACE TABLE pairs AS
        SELECT a, b, list(DISTINCT rule ORDER BY rule) AS rules FROM candidates GROUP BY a, b
    """)
    m, = con.execute("SELECT count(*) FROM pairs").fetchone()
    print(f"candidate pairs after union: {m:,} ({m / (n * (n - 1) / 2):.2e} of all pairs)")


if __name__ == "__main__":
    main()
