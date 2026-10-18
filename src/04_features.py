"""Compute pairwise comparison features for every candidate pair."""
from common import connect


def main() -> None:
    con = connect()
    # How common a name / address is: generic names and shared office buildings
    # need more evidence before two loans can be called the same business.
    con.execute("""
        CREATE OR REPLACE TABLE name_freq AS
        SELECT name_std, count(*) AS n FROM borrowers GROUP BY 1
    """)
    con.execute("""
        CREATE OR REPLACE TABLE addr_freq AS
        SELECT zip5, addr_std, count(*) AS n FROM borrowers WHERE addr_std <> '' GROUP BY 1, 2
    """)
    con.execute("""
        CREATE OR REPLACE TABLE features AS
        SELECT
            p.a, p.b, p.rules,
            jaro_winkler_similarity(x.name_std, y.name_std) AS name_jw,
            1 - levenshtein(x.name_std, y.name_std) / greatest(length(x.name_std), length(y.name_std), 1) AS name_lev,
            len(list_intersect(string_split(x.name_std, ' '), string_split(y.name_std, ' ')))
              / greatest(len(list_distinct(list_concat(string_split(x.name_std, ' '), string_split(y.name_std, ' ')))), 1) AS name_tok_jacc,
            -- best similarity across legal and DBA names (A legal/DBA vs B legal/DBA)
            greatest(
                jaro_winkler_similarity(x.name_std, y.name_std),
                CASE WHEN x.dba_std <> '' THEN jaro_winkler_similarity(x.dba_std, y.name_std) ELSE 0 END,
                CASE WHEN y.dba_std <> '' THEN jaro_winkler_similarity(x.name_std, y.dba_std) ELSE 0 END,
                CASE WHEN x.dba_std <> '' AND y.dba_std <> '' THEN jaro_winkler_similarity(x.dba_std, y.dba_std) ELSE 0 END
            ) AS name_best_jw,
            len(list_intersect(string_split(trim(x.name_std || ' ' || x.dba_std), ' '), string_split(trim(y.name_std || ' ' || y.dba_std), ' ')))
              / greatest(len(list_distinct(list_concat(string_split(trim(x.name_std || ' ' || x.dba_std), ' '), string_split(trim(y.name_std || ' ' || y.dba_std), ' ')))), 1) AS name_all_tok_jacc,
            (x.name_std = y.name_std)::INT AS name_eq,
            (x.dba_std <> '' AND (x.dba_std = y.name_std OR x.dba_std = y.dba_std)
              OR y.dba_std <> '' AND y.dba_std = x.name_std)::INT AS dba_cross,
            jaro_winkler_similarity(x.addr_std, y.addr_std) AS addr_jw,
            (x.addr_std = y.addr_std AND x.addr_std <> '')::INT AS addr_eq,
            (x.house_no = y.house_no AND x.house_no <> '')::INT AS house_eq,
            jaro_winkler_similarity(x.street, y.street) AS street_jw,
            (x.zip5 = y.zip5)::INT AS zip_eq,
            (x.city = y.city)::INT AS city_eq,
            (x.naics = y.naics)::INT AS naics6_eq,
            (left(x.naics, 3) = left(y.naics, 3))::INT AS naics3_eq,
            (x.business_type = y.business_type)::INT AS btype_eq,
            (x.business_type IN ('Sole Proprietorship', 'Self-Employed Individuals', 'Independent Contractors'))::INT
              + (y.business_type IN ('Sole Proprietorship', 'Self-Employed Individuals', 'Independent Contractors'))::INT AS n_individual,
            (x.franchise IS NOT NULL OR y.franchise IS NOT NULL)::INT AS any_franchise,
            (x.franchise IS NOT NULL AND x.franchise = y.franchise)::INT AS franchise_eq,
            (x.lender = y.lender)::INT AS same_lender,
            least(x.amount, y.amount) / greatest(x.amount, y.amount, 1) AS amount_ratio,
            least(x.jobs, y.jobs) / greatest(x.jobs, y.jobs, 1) AS jobs_ratio,
            abs(date_diff('day', x.approved, y.approved)) AS days_apart,
            least(length(x.name_std), length(y.name_std)) AS name_len,
            ln(greatest(nx.n, ny.n)) AS log_name_freq,
            ln(greatest(coalesce(ax.n, 1), coalesce(ay.n, 1))) AS log_addr_freq
        FROM pairs p
        JOIN borrowers x ON x.loan_id = p.a
        JOIN borrowers y ON y.loan_id = p.b
        JOIN name_freq nx ON nx.name_std = x.name_std
        JOIN name_freq ny ON ny.name_std = y.name_std
        LEFT JOIN addr_freq ax ON ax.zip5 = x.zip5 AND ax.addr_std = x.addr_std
        LEFT JOIN addr_freq ay ON ay.zip5 = y.zip5 AND ay.addr_std = y.addr_std
    """)
    n, = con.execute("SELECT count(*) FROM features").fetchone()
    print(f"features for {n:,} pairs")
    print(con.execute("SELECT * FROM features USING SAMPLE 3").df().T.to_string())


if __name__ == "__main__":
    main()
