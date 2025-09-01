# free_float.py
import warnings
import pandas as pd
import yfinance as yf

from data.fetch_data import download_prices  # reuse price loader

# --------- Free Float Data ---------

def fetch_free_float_table(tickers):
    """
    Build a table with implied shares outstanding (B), free float (B),
    free float %, and validity.

    Returns DataFrame with:
      ['Ticker','Implied Shares Outstanding (B)','Free Float (B)','Free Float %','Valid Data']
    """
    results = []

    for symbol in tickers:
        t = yf.Ticker(symbol)
        info = t.info
        market_cap = info.get("marketCap")
        current_price = info.get("currentPrice")
        float_shares = info.get("floatShares")

        implied_shares = (market_cap / current_price) if (market_cap and current_price) else None
        implied_shares_b = round(implied_shares / 1e9, 3) if implied_shares else None
        free_float_b = round(float_shares / 1e9, 3) if float_shares else None

        if implied_shares and float_shares:
            free_float_pct = round((float_shares / implied_shares) * 100, 2)
            valid = float_shares <= implied_shares
        else:
            free_float_pct = None
            valid = False

        results.append({
            "Ticker": symbol,
            "Implied Shares Outstanding (B)": implied_shares_b,
            "Free Float (B)": free_float_b,
            "Free Float %": free_float_pct,
            "Valid Data": valid
        })
    return pd.DataFrame(results)


def add_free_float_weights(df):
    """
    Append 'Free Float Weight' = Free Float (B) / sum(Free Float (B)).
    """
    df = df.copy()
    total = df["Free Float (B)"].sum(min_count=1)
    if total and total > 0:
        df["Free Float Weight"] = df["Free Float (B)"].apply(
            lambda x: round(x / total, 6) if pd.notnull(x) else None
        )
    else:
        df["Free Float Weight"] = None
    return df


# --------- Portfolio Simulation ---------

def simulate_portfolio_from_weights(prices, weights, initial_investment=1000.0):
    """
    Given a price DataFrame (columns=tickers) and a weight mapping (ticker->weight),
    compute daily percentage changes and cumulative portfolio value:

      V_t = V_{t-1} * (1 + sum_i w_i * r_{i,t})
    """
    w = pd.Series(weights, dtype=float).reindex(prices.columns).fillna(0.0)
    pct_change = prices.pct_change().fillna(0.0)

    values = [initial_investment]
    for i in range(1, len(pct_change)):
        daily_ret = float((pct_change.iloc[i] * w).sum())
        values.append(values[-1] * (1.0 + daily_ret))

    out = pct_change.copy()
    out["Portfolio Value"] = values
    return out


def build_portfolio_from_free_float(tickers, start, end, initial_investment=1000.0,
                                    auto_adjust=True, price_field="Close"):
    """
    Pipeline:
      1) Fetch free-float table + weights
      2) Download prices (from fetch_data.py)
      3) Simulate portfolio using those weights

    Returns: (df_weights, df_portfolio)
    """
    df_weights = fetch_free_float_table(tickers)
    df_weights = add_free_float_weights(df_weights)

    valid = df_weights["Valid Data"] & df_weights["Free Float Weight"].notnull()
    if not valid.any():
        warnings.warn("No valid free-float weights available; portfolio cannot be simulated.")
        prices = download_prices(
            tickers, start=start, end=end, auto_adjust=auto_adjust, price_field=price_field
        )
        empty = prices.pct_change().fillna(0.0)
        empty["Portfolio Value"] = initial_investment
        return df_weights, empty

    weights = df_weights.loc[valid].set_index("Ticker")["Free Float Weight"].to_dict()

    prices = download_prices(
        list(weights.keys()), start=start, end=end, auto_adjust=auto_adjust, price_field=price_field
    )

    df_port = simulate_portfolio_from_weights(
        prices, weights, initial_investment=initial_investment
    )

    return df_weights, df_port
