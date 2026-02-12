"""HDF5 and SQLite storage backends for market data."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from config.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent / "cache"


class HDF5Store:
    """HDF5-based storage using PyTables."""

    def __init__(self, path: str | Path | None = None):
        if path is None:
            DEFAULT_DB_DIR.mkdir(exist_ok=True)
            path = DEFAULT_DB_DIR / "market_data.h5"
        self.path = Path(path)

    def write(self, key: str, df: pd.DataFrame) -> None:
        """Write DataFrame to HDF5 store."""
        df.to_hdf(str(self.path), key=key, mode="a", complevel=9, complib="blosc")
        logger.info("hdf5_write", key=key, rows=len(df))

    def read(self, key: str) -> pd.DataFrame:
        """Read DataFrame from HDF5 store."""
        df = pd.read_hdf(str(self.path), key=key)
        logger.info("hdf5_read", key=key, rows=len(df))
        return df

    def keys(self) -> list[str]:
        """List all keys in the HDF5 store."""
        with pd.HDFStore(str(self.path), mode="r") as store:
            return store.keys()

    def delete(self, key: str) -> None:
        """Remove a dataset from the HDF5 store."""
        with pd.HDFStore(str(self.path), mode="a") as store:
            store.remove(key)
        logger.info("hdf5_delete", key=key)


class SQLiteStore:
    """SQLite-based storage for structured market data."""

    def __init__(self, path: str | Path | None = None):
        if path is None:
            DEFAULT_DB_DIR.mkdir(exist_ok=True)
            path = DEFAULT_DB_DIR / "market_data.db"
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path))

    def write(self, table: str, df: pd.DataFrame, if_exists: str = "replace") -> None:
        """Write DataFrame to SQLite table."""
        with self._connect() as conn:
            df.to_sql(table, conn, if_exists=if_exists, index=True)
        logger.info("sqlite_write", table=table, rows=len(df))

    def read(self, table: str) -> pd.DataFrame:
        """Read DataFrame from SQLite table."""
        with self._connect() as conn:
            df = pd.read_sql(f"SELECT * FROM [{table}]", conn, index_col="Date",
                             parse_dates=["Date"])
        logger.info("sqlite_read", table=table, rows=len(df))
        return df

    def tables(self) -> list[str]:
        """List all tables in the SQLite database."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            return [row[0] for row in cursor.fetchall()]

    def delete(self, table: str) -> None:
        """Drop a table from the SQLite database."""
        with self._connect() as conn:
            conn.execute(f"DROP TABLE IF EXISTS [{table}]")
        logger.info("sqlite_delete", table=table)
