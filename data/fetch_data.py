# data/fetch_data.py
"""
Data fetching utilities for the alpha-models package.

Currently supports:
- Yahoo Finance (via yfinance) for adjusted closing prices

Future extensions:
- Vendor APIs (Polygon, Quandl, Refinitiv, Bloomberg, etc.)
- Fundamental data
- Alternative data (sentiment, macro series)
"""

from __future__ import annotations
import pandas as pd
import yfinance as yf


def load_prices(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted close prices from Yahoo Finance.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols (e.g., ["AAPL", "MSFT", "GOOG"]).
    start : str
        Start date (YYYY-MM-DD).
    end : str or None
        End date (YYYY-MM-DD). If None, fetches until the latest.
    auto_adjust : bool
        Whether to adjust prices for splits/dividends.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with dates as index and tickers as columns.
    """
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
    )["Close"]

    if isinstance(data, pd.Series):  # single ticker case
        data = data.to_frame(name=tickers[0])

    return data.dropna(how="all").sort_index()
