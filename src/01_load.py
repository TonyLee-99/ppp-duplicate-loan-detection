"""Download the SBA PPP FOIA loan-level files and load them into DuckDB.

Source: https://data.sba.gov/dataset/ppp-foia (data as of 2024-09-30, public domain).
"""
import subprocess

from common import DATA, SBA_BASE, SBA_FILES, connect

RAW = DATA / "raw"


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name in SBA_FILES:
        out = RAW / f"{name}.csv"
        if out.exists() and out.stat().st_size > 0:
            continue
        print(f"downloading {name}")
        subprocess.run(["curl", "-sL", "--retry", "3", "-o", str(out), f"{SBA_BASE}/{name}.csv"], check=True)


def load() -> None:
    con = connect()
    # A few rows contain non-UTF-8 bytes, so read as Latin-1.
    con.execute(f"""
        CREATE OR REPLACE TABLE loans AS
        SELECT * FROM read_csv('{RAW}/*.csv', header=true, all_varchar=true,
                               union_by_name=true, encoding='latin-1')
    """)
    n, = con.execute("SELECT count(*) FROM loans").fetchone()
    print(f"loaded {n:,} loans")


if __name__ == "__main__":
    download()
    load()
