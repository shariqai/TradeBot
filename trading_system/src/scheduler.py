from apscheduler.schedulers.blocking import BlockingScheduler
import logging
from data_acquisition import fetch_stock_data
from trading_strategies import momentum_trading
from execution import execute_trade

def trading_bot():
    """
    Main trading bot function.
    """
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    for symbol in symbols:
        data = fetch_stock_data(symbol, start='2022-01-01', end='2022-12-31')
        if data is not None and 'Close' in data.columns:
            momentum_signal = momentum_trading(data['Close'])
            if momentum_signal.iloc[-1] > 0:
                execute_trade(symbol, 'buy', 10)
            elif momentum_signal.iloc[-1] < 0:
                execute_trade(symbol, 'sell', 10)

# Set up scheduler
scheduler = BlockingScheduler()
scheduler.add_job(trading_bot, 'interval', hours=1)

if __name__ == "__main__":
    try:
        logging.info("Starting trading bot...")
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Stopping trading bot...")
        scheduler.shutdown()
