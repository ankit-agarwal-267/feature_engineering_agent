import pandas as pd
import polars as pl
from typing import Any, Dict, Union
import sqlalchemy as sa
from fe_agent.ingestion.base import AbstractDataLoader
from fe_agent.profiler.semantic_types import RawDataFrame

class SQLLoader(AbstractDataLoader):
    def supports(self, source: str | dict) -> bool:
        if isinstance(source, str):
            # Very basic check for DSN format or starts with standard prefixes
            prefixes = ('postgresql', 'mysql', 'sqlite', 'mssql', 'oracle', 'mariadb')
            return any(source.startswith(p) for p in prefixes)
        return False

    def load(self, source: str | dict, **kwargs: Any) -> RawDataFrame:
        if not isinstance(source, str):
            raise ValueError("SQLLoader requires a connection URL string.")

        query = kwargs.get('sql_query')
        table = kwargs.get('sql_table')
        limit = kwargs.get('limit')
        
        if not query and not table:
            raise ValueError("Either 'sql_query' or 'sql_table' must be provided for SQLLoader.")

        if table and not query:
            query = f"SELECT * FROM {table}"
            if limit:
                query += f" LIMIT {limit}"

        backend = kwargs.get('backend', 'pandas')
        engine = sa.create_engine(source)
        
        if backend == 'pandas':
            with engine.connect() as conn:
                return pd.read_sql(query, conn)
        elif backend == 'polars':
            # Polars read_database uses connectorx or adbc, but we can use sqlalchemy as a fallback
            return pl.read_database(query, source)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    @staticmethod
    def mask_sql_credentials(url: str) -> str:
        """
        Utility to mask credentials in a SQL URL for safe logging.
        """
        try:
            res = sa.engine.url.make_url(url)
            return res.render_as_string(hide_password=True)
        except Exception:
            return "masked_url"
