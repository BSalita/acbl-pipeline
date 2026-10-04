"""Read ACBL SQLite into Polars.

Windows application control blocks the ADBC driver
(libadbc_driver_sqlite.so, LoadLibrary error 4551). DuckDB's SQLite
scanner is already allowed, so stage 2 uses it.
"""

from __future__ import annotations

import duckdb
import polars as pl

_TRUE = ("1", "true", "t")
_FALSE = ("0", "false", "f")


def read_sqlite_query(
    uri: str,
    query: str,
    schema_overrides: dict[str, pl.DataType] | None = None,
) -> pl.DataFrame:
    """Run one SELECT against a sqlite:/// URI and apply column types."""
    path = uri.removeprefix("sqlite:///").replace("\\", "/")
    path_sql = path.replace("'", "''")
    con = duckdb.connect()
    try:
        con.execute("LOAD sqlite")
        # Keep mixed text such as hand_record_id 'SHUFFLE' instead of
        # letting DuckDB guess an integer column and drop the text.
        con.execute("SET sqlite_all_varchar = true")
        con.execute(f"ATTACH '{path_sql}' AS src (TYPE SQLITE, READ_ONLY)")
        con.execute("SET search_path = 'src'")
        frame = con.execute(query).pl()
    finally:
        con.close()
    if not schema_overrides:
        return frame
    missing = [name for name in schema_overrides if name not in frame.columns]
    if missing:
        raise KeyError(f"SQLite query is missing columns: {missing}")
    return frame.with_columns(
        [_cast_column(name, dtype) for name, dtype in schema_overrides.items()]
    )


def _cast_column(name: str, dtype: pl.DataType) -> pl.Expr:
    col = pl.col(name)
    if dtype == pl.Boolean:
        lowered = col.str.to_lowercase()
        return (
            pl.when(col.is_null() | (col == ""))
            .then(None)
            .when(lowered.is_in(list(_TRUE)))
            .then(pl.lit(True))
            .when(lowered.is_in(list(_FALSE)))
            .then(pl.lit(False))
            .otherwise(None)
            .alias(name)
        )
    if dtype == pl.String:
        return col.alias(name)
    return col.cast(dtype, strict=False).alias(name)
