import numpy as np
import pandas as pd

def pairs_trading(stock1, stock2, data):
    """
    Pairs trading strategy.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")
    
    if stock1 in data.columns and stock2 in data.columns:
        spread = data[stock1] - data[stock2]
        mean = spread.mean()
        std_dev = spread.std()
        if std_dev == 0:
            return 0  # Avoid division by zero
        return (spread.iloc[-1] - mean) / std_dev
    else:
        raise ValueError(f"Columns for '{stock1}' and/or '{stock2}' not found in data.")

def momentum_trading(data, lookback=14):
    """
    Momentum trading strategy.
    """
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError("Data must be a pandas Series or DataFrame.")
    
    if len(data) > lookback:
        momentum = data - data.shift(lookback)
        return momentum.dropna()  # Drop NaN values resulting from the shift
    else:
        raise ValueError("Data length must be greater than the lookback period.")

def mean_reversion(data, lookback=14):
    """
    Mean reversion strategy.
    """
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError("Data must be a pandas Series or DataFrame.")
    
    if len(data) >= lookback:
        moving_avg = data.rolling(window=lookback).mean()
        return (data - moving_avg).dropna()
    else:
        raise ValueError("Data length must be greater than or equal to the lookback period.")
