import pandas as pd

def calculate_correlations(prices, df_portfolio=None):
    """
    Calculate correlation matrix of daily returns between tickers,
    and optionally include the portfolio value returns.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of closing prices (columns = tickers).
    df_portfolio : pd.DataFrame or None
        DataFrame containing 'Portfolio Value' (optional).

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """
    # Daily returns of tickers
    returns = prices.pct_change().dropna()
    
    # If portfolio DataFrame is provided, add portfolio returns
    if df_portfolio is not None and "Portfolio Value" in df_portfolio:
        portfolio_returns = df_portfolio["Portfolio Value"].pct_change().dropna()
        returns = returns.join(portfolio_returns.rename("Portfolio"))

    # Correlation matrix
    return returns.corr()


def calculate_annualized_volatility(prices, df_portfolio=None):
    """
    Calculate annualized volatility of daily returns for each ticker,
    and optionally include the portfolio value returns.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of closing prices (columns = tickers).
    df_portfolio : pd.DataFrame or None
        DataFrame containing 'Portfolio Value' (optional).

    Returns
    -------
    pd.Series
        Annualized volatility for each ticker and portfolio (if provided).
    """
    # Daily returns of tickers
    returns = prices.pct_change().dropna()

    # If portfolio DataFrame is provided, add portfolio returns
    if df_portfolio is not None and "Portfolio Value" in df_portfolio:
        portfolio_returns = df_portfolio["Portfolio Value"].pct_change().dropna()
        returns = returns.join(portfolio_returns.rename("Portfolio"))

    # Annualized volatility calculation
    annualized_volatility = returns.std() * (252 ** 0.5)
    
    return annualized_volatility