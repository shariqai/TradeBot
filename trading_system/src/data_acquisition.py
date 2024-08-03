# src/data_acquisition.py
import pandas as pd
import numpy as np
import yfinance as yf

def fetch_stock_data(symbol, start, end, interval="1d"):
    """
    Fetch historical stock data from Yahoo Finance.
    """
    stock = yf.download(symbol, start=start, end=end, interval=interval)
    return stock

def fetch_option_data(symbol, expiry):
    """
    Fetch option data for a given stock and expiry date.
    """
    # Example using yfinance for simplicity
    stock = yf.Ticker(symbol)
    options = stock.option_chain(expiry)
    return options.calls, options.puts
