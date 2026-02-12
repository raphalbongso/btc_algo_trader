"""Tests for data.storage."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from data.storage import SQLiteStore


class TestSQLiteStore:
    @pytest.fixture
    def store(self, tmp_path):
        return SQLiteStore(path=tmp_path / "test.db")

    @pytest.fixture
    def sample_df(self, btc_data):
        return btc_data.head(50)

    def test_write_and_read(self, store, sample_df):
        store.write("btc", sample_df)
        result = store.read("btc")
        assert len(result) == len(sample_df)
        assert set(result.columns) == set(sample_df.columns)

    def test_tables(self, store, sample_df):
        store.write("btc", sample_df)
        store.write("eth", sample_df)
        tables = store.tables()
        assert "btc" in tables
        assert "eth" in tables

    def test_delete(self, store, sample_df):
        store.write("btc", sample_df)
        store.delete("btc")
        assert "btc" not in store.tables()

    def test_overwrite(self, store, sample_df):
        store.write("btc", sample_df)
        shorter = sample_df.head(10)
        store.write("btc", shorter)
        result = store.read("btc")
        assert len(result) == 10
