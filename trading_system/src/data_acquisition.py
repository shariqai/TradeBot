import pandas as pd
import yfinance as yf

def fetch_stock_data(symbol, start, end, interval="1d"):
    """
    Fetch historical stock data from Yahoo Finance.
    """
    try:
        stock = yf.download(symbol, start=start, end=end, interval=interval)
        return stock
    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {e}")
        return None

def fetch_option_data(symbol, expiry):
    """
    Fetch option data for a given stock and expiry date.
    """
    try:
        stock = yf.Ticker(symbol)
        options = stock.option_chain(expiry)
        return options.calls, options.puts
    except Exception as e:
        print(f"Error fetching option data for {symbol}: {e}")
        return None, None
