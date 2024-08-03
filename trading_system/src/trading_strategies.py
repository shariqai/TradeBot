# src/trading_strategies.py
import numpy as np

def pairs_trading(stock1, stock2, data):
    """
    Pairs trading strategy.
    """
    spread = data[stock1] - data[stock2]
    mean = np.mean(spread)
    std_dev = np.std(spread)
    return (spread[-1] - mean) / std_dev

def momentum_trading(data, lookback=14):
    """
    Momentum trading strategy.
    """
    momentum = data - data.shift(lookback)
    return momentum

def mean_reversion(data, lookback=14):
    """
    Mean reversion strategy.
    """
    moving_avg = data.rolling(window=lookback).mean()
    return data - moving_avg
