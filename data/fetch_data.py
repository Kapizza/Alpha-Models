# fetch_data.py
import pandas as pd
import yfinance as yf

def download_prices(tickers, start, end, auto_adjust=True, price_field="Close"):
    """
    Download OHLCV (or Close) data for multiple tickers and return a DataFrame
    of the selected price_field (e.g., 'Close', 'Adj Close').
    """
    tickers = list(tickers)
    data = yf.download(
        tickers,
        start=start,
        end=end,            # end is exclusive
        auto_adjust=auto_adjust,
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        if price_field not in data.columns.levels[0]:
            raise ValueError(f"Requested price_field '{price_field}' not found.")
        prices = data[price_field].copy()
    else:
        if price_field not in data.columns:
            raise ValueError(f"Requested price_field '{price_field}' not found.")
        prices = data[[price_field]].copy()
        prices.columns = [tickers[0]]

    existing = [c for c in tickers if c in prices.columns]
    prices = prices.reindex(columns=existing).dropna(how="all").ffill()
    return prices


def load_prices(tickers, start=None, end=None, auto_adjust=True):
    """
    Download (adjusted or raw per auto_adjust) Close prices from Yahoo Finance.
    Returns a wide DataFrame (index=dates, columns=tickers).
    """
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
    )["Close"]

    if isinstance(data, pd.Series):  # single ticker case
        data = data.to_frame(name=tickers[0] if isinstance(tickers, (list, tuple)) else str(tickers))

    return data.dropna(how="all").sort_index()
