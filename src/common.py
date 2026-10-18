"""Shared paths and helpers."""
from pathlib import Path
import os

import duckdb

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PPP_DATA", REPO.parent / "ppp-data"))
DB_PATH = DATA / "ppp.duckdb"
LABELS = REPO / "labels"
REPORTS = REPO / "reports"
FIGS = REPORTS / "figures"

SBA_BASE = "https://data.sba.gov/sites/default/files/distribution/SBA-OCA-2022-07-001"
SBA_FILES = ["public_150k_plus_240930"] + [f"public_up_to_150k_{i}_240930" for i in range(1, 13)]


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    DATA.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH), read_only=read_only)
    con.execute("SET memory_limit='10GB'")
    con.execute("SET preserve_insertion_order=false")
    return con
