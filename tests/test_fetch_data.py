# tests/test_fetch_data.py

import pandas as pd
from data.fetch_data import load_prices

def _all_numeric_dtypes(df: pd.DataFrame) -> bool:
    return df.dtypes.apply(pd.api.types.is_numeric_dtype).all()

def test_load_prices_single_and_multi():
    # Single ticker
    df_single = load_prices(["AAPL"], start="2022-01-01", end="2022-02-01")
    assert list(df_single.columns) == ["AAPL"]
    assert isinstance(df_single.index, pd.DatetimeIndex)
    assert _all_numeric_dtypes(df_single)

    # Multiple tickers
    df_multi = load_prices(["AAPL", "MSFT"], start="2022-01-01", end="2022-02-01")
    assert set(df_multi.columns) == {"AAPL", "MSFT"}
    assert isinstance(df_multi.index, pd.DatetimeIndex)
    assert _all_numeric_dtypes(df_multi)

def test_single_vs_multiple_consistency():
    df_single = load_prices(["AAPL"], start="2022-01-01", end="2022-02-01")
    df_multi = load_prices(["AAPL", "MSFT"], start="2022-01-01", end="2022-02-01")
    pd.testing.assert_series_equal(df_single["AAPL"], df_multi["AAPL"], check_names=False)
