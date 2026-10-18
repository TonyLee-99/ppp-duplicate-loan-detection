"""Standardize borrower names and addresses for first-draw (PPP) loans.

Output table: borrowers (one row per first-draw loan) with cleaned keys used for
blocking and comparison.
"""
from common import connect

LEGAL = r"\b(L ?L ?C|INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|P ?L ?L ?C|P ?C|P ?A|L ?L ?P|L ?P|THE|PLC|LC)\b"

# Common USPS street suffix and direction abbreviations.
ADDR_SUBS = [
    ("STREET", "ST"), ("AVENUE", "AVE"), ("AV", "AVE"), ("ROAD", "RD"), ("DRIVE", "DR"),
    ("BOULEVARD", "BLVD"), ("LANE", "LN"), ("COURT", "CT"), ("PLACE", "PL"), ("HIGHWAY", "HWY"),
    ("PARKWAY", "PKWY"), ("CIRCLE", "CIR"), ("TERRACE", "TER"), ("TRAIL", "TRL"), ("SQUARE", "SQ"),
    ("NORTH", "N"), ("SOUTH", "S"), ("EAST", "E"), ("WEST", "W"),
    ("NORTHEAST", "NE"), ("NORTHWEST", "NW"), ("SOUTHEAST", "SE"), ("SOUTHWEST", "SW"),
]


def addr_expr(col: str) -> str:
    e = f"upper(coalesce({col}, ''))"
    e = f"regexp_replace({e}, '[.,]', ' ', 'g')"
    # Drop unit designators and the token that follows them.
    e = f"regexp_replace({e}, '\\b(STE|SUITE|UNIT|APT|APARTMENT|FL|FLOOR|RM|ROOM|BLDG|BUILDING|LOT|SPC|SPACE|DEPT)\\b\\s*\\S*', ' ', 'g')"
    e = f"regexp_replace({e}, '#\\s*\\S*', ' ', 'g')"
    for full, abbr in ADDR_SUBS:
        e = f"regexp_replace({e}, '\\b{full}\\b', '{abbr}', 'g')"
    e = f"regexp_replace({e}, '[^A-Z0-9 ]', ' ', 'g')"
    return f"trim(regexp_replace({e}, '\\s+', ' ', 'g'))"


def name_expr(col: str) -> str:
    e = f"upper(coalesce({col}, ''))"
    e = f"replace(replace(replace({e}, '&#39;', ''''), '&AMP;', '&'), '&QUOT;', '')"
    e = f"replace({e}, '&', ' AND ')"
    e = f"regexp_replace({e}, '[^A-Z0-9 ]', ' ', 'g')"
    e = f"regexp_replace({e}, '{LEGAL}', ' ', 'g')"
    return f"trim(regexp_replace({e}, '\\s+', ' ', 'g'))"


def main() -> None:
    con = connect()
    raw_name = "upper(coalesce(BorrowerName, ''))"
    dba_split = f"regexp_extract({raw_name}, '^(.*?)\\s+(D/?B/?A/?|DOING BUSINESS AS)\\s+(.*)$', ['legal', 'sep', 'dba'])"
    con.execute(f"""
        CREATE OR REPLACE TABLE borrowers AS
        WITH base AS (
            SELECT
                LoanNumber AS loan_id,
                BorrowerName AS raw_name,
                BorrowerAddress AS raw_address,
                upper(BorrowerCity) AS city,
                upper(BorrowerState) AS state,
                left(BorrowerZip, 5) AS zip5,
                try_strptime(DateApproved, '%m/%d/%Y')::DATE AS approved,
                LoanStatus AS status,
                try_strptime(LoanStatusDate, '%m/%d/%Y')::DATE AS status_date,
                try_cast(CurrentApprovalAmount AS DOUBLE) AS amount,
                try_cast(JobsReported AS INTEGER) AS jobs,
                NAICSCode AS naics,
                BusinessType AS business_type,
                FranchiseName AS franchise,
                OriginatingLender AS lender,
                {dba_split} AS dba_parts
            FROM loans
            WHERE ProcessingMethod = 'PPP'
              AND BorrowerName IS NOT NULL
              AND BorrowerName NOT ILIKE 'Exemption%'
              AND upper(BorrowerName) NOT IN ('NOT AVAILABLE', 'N/A')
        )
        SELECT * EXCLUDE (dba_parts),
            {name_expr("CASE WHEN dba_parts.legal <> '' THEN dba_parts.legal ELSE raw_name END")} AS name_std,
            {name_expr("dba_parts.dba")} AS dba_std,
            {addr_expr("raw_address")} AS addr_std
        FROM base
    """)
    con.execute("""
        CREATE OR REPLACE TABLE borrowers AS
        SELECT *,
            regexp_extract(addr_std, '^([0-9]+)', 1) AS house_no,
            trim(regexp_replace(addr_std, '^[0-9]+\\s*', '')) AS street,
            split_part(name_std, ' ', 1) AS name_tok1,
            left(regexp_replace(name_std, ' ', '', 'g'), 6) AS name_pre6
        FROM borrowers
    """)
    n, = con.execute("SELECT count(*) FROM borrowers").fetchone()
    print(f"{n:,} first-draw borrowers standardized")
    print(con.execute("SELECT raw_name, name_std, dba_std, raw_address, addr_std, house_no, street FROM borrowers USING SAMPLE 8").df().to_string())


if __name__ == "__main__":
    main()
