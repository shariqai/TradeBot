import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from data_acquisition import fetch_stock_data
from trading_strategies import momentum_trading
from execution import execute_trade

# Set up logging
logging.basicConfig(level=logging.INFO)

def trading_bot():
    logging.info("Running trading bot...")
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    for symbol in symbols:
        data = fetch_stock_data(symbol, start='2022-01-01', end='2022-12-31')
        if data is not None and 'Close' in data.columns:
            momentum_signal = momentum_trading(data['Close'])
            if momentum_signal.iloc[-1] > 0:
                execute_trade(symbol, 'buy', 10)
                logging.info(f"Executed buy for {symbol}")
            elif momentum_signal.iloc[-1] < 0:
                execute_trade(symbol, 'sell', 10)
                logging.info(f"Executed sell for {symbol}")

# Set up scheduler
scheduler = BlockingScheduler()
scheduler.add_job(trading_bot, 'interval', hours=1)

if __name__ == "__main__":
    logging.info("Starting trading bot scheduler...")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Stopping trading bot scheduler...")
        scheduler.shutdown()
