import numpy as np
import pandas as pd
import requests
import time
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import psycopg2
from psycopg2 import sql

# Twilio and email configuration
TWILIO_ACCOUNT_SID = 'your_twilio_account_sid'
TWILIO_AUTH_TOKEN = 'your_twilio_auth_token'
TWILIO_PHONE_NUMBER = 'your_twilio_phone_number'
TO_PHONE_NUMBER = '16465587623'
EMAIL_ADDRESS = 'shrqshhb@gmail.com'
EMAIL_PASSWORD = 'your_email_password'

# PostgreSQL database configuration
DB_HOST = 'your_db_host'
DB_PORT = 'your_db_port'
DB_NAME = 'your_db_name'
DB_USER = 'your_db_user'
DB_PASSWORD = 'your_db_password'

# Connect to PostgreSQL database
def connect_to_db():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

# Function to create table in the database
def create_table():
    conn = connect_to_db()
    if conn:
        cursor = conn.cursor()
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS stock_recommendations (
            id SERIAL PRIMARY KEY,
            stock VARCHAR(50),
            signal VARCHAR(50),
            expected_return FLOAT,
            justification TEXT,
            risk_assessment TEXT,
            volume INT,
            market_impact FLOAT,
            timestamp TIMESTAMP,
            sec_compliance BOOLEAN DEFAULT TRUE,
            regulation_remarks TEXT
        );
        '''
        cursor.execute(create_table_query)
        conn.commit()
        cursor.close()
        conn.close()

# Function to save CSV data to the database
def save_csv_to_db(data):
    conn = connect_to_db()
    if conn:
        cursor = conn.cursor()
        insert_query = '''
        INSERT INTO stock_recommendations (stock, signal, expected_return, justification, risk_assessment, volume, market_impact, timestamp, sec_compliance, regulation_remarks)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        '''
        cursor.executemany(insert_query, data)
        conn.commit()
        cursor.close()
        conn.close()

# Define individual strategies and functions
def latency_arbitrage(order_book, latency_data):
    # Implement latency arbitrage strategy
    latency_diff = np.random.uniform(0, 0.1)
    arbitrage_opportunity = order_book['ask'] - (order_book['bid'] + latency_diff)
    return arbitrage_opportunity

def market_microstructure_analysis(order_book):
    # Implement market microstructure analysis for HFT
    spread = order_book['ask'] - order_book['bid']
    depth = (order_book['ask_size'] + order_book['bid_size']) / 2
    microstructure_metrics = {'spread': spread, 'depth': depth}
    return microstructure_metrics

def quote_stuffing_detection(order_book):
    # Implement detection of quote stuffing
    order_rate = len(order_book) / (order_book['time'].max() - order_book['time'].min())
    stuffing_detected = order_rate > 1000
    return stuffing_detected

def momentum_ignition(order_book, volume, momentum_threshold):
    # Implement momentum ignition strategy
    if volume > momentum_threshold:
        return 'Ignite Momentum'
    return 'No Action'

def spoofing_detection(order_book):
    # Implement detection of spoofing activities
    bid_ask_spread = order_book['ask'] - order_book['bid']
    spoofing_detected = bid_ask_spread > np.percentile(bid_ask_spread, 95)
    return spoofing_detected

def order_anticipation(order_book, order_flow_data):
    # Implement order anticipation strategy
    large_order_threshold = np.percentile(order_flow_data['size'], 95)
    anticipation_signal = order_flow_data['size'] > large_order_threshold
    return anticipation_signal

def order_book_imbalance(order_book):
    # Implement order book imbalance analysis for HFT
    imbalance = order_book['ask_size'] - order_book['bid_size']
    imbalance_signal = 'Buy' if imbalance < 0 else 'Sell'
    return imbalance_signal

def flash_ordering(order_book, latency_data):
    # Implement flash ordering strategy
    latency_advantage = np.random.uniform(0, 0.05)
    best_price = min(order_book['ask']) - latency_advantage
    return best_price

def passive_aggressive_order_execution(order_book, target_price, volume):
    # Implement passive-aggressive order execution strategy
    if target_price < order_book['ask']:
        return {'action': 'Aggressive Buy', 'volume': volume}
    else:
        return {'action': 'Passive Wait', 'volume': 0}

def optimal_liquidity_consumption(order_book, volume):
    # Implement optimal liquidity consumption strategy
    depth = order_book['ask_size'].cumsum()
    optimal_depth = depth[depth >= volume].min()
    optimal_price = order_book[order_book['ask_size'].cumsum() == optimal_depth]['ask']
    return optimal_price

def multi_leg_arbitrage(opportunity_data):
    # Implement multi-leg arbitrage strategy
    legs = opportunity_data['legs']
    profits = []
    for leg in legs:
        profit = leg['sell_price'] - leg['buy_price']
        profits.append(profit)
    return sum(profits)

def high_frequency_mean_reversion(price_data, window_size):
    # Implement high-frequency mean reversion strategy
    mean_price = price_data.rolling(window=window_size).mean()
    return (price_data - mean_price).abs().idxmin()

def colocation_arbitrage(latency_data, server_locations):
    # Implement colocation arbitrage strategy
    best_location = server_locations[np.argmin(latency_data)]
    return best_location

def liquidity_detection(order_book_data):
    # Detect liquidity pools for strategic order placement
    liquidity_pools = order_book_data['ask_size'] + order_book_data['bid_size']
    high_liquidity_zones = liquidity_pools[liquidity_pools > np.percentile(liquidity_pools, 95)]
    return high_liquidity_zones

def adaptive_hft_strategy(market_conditions, historical_data):
    # Adapt HFT strategy based on market conditions
    volatility = np.std(historical_data)
    if market_conditions['volatility'] > volatility:
        return 'High Volatility Strategy'
    else:
        return 'Low Volatility Strategy'

def market_manipulation_detection(order_book_data, trade_data):
    # Detect market manipulation activities such as layering or spoofing
    layering = (order_book_data['ask_size'] > 1000) & (order_book_data['bid_size'] > 1000)
    manipulation_detected = any(layering)
    return manipulation_detected

def flash_order_detection(order_data, timing_data):
    # Detect and analyze the presence of flash orders
    flash_order_events = (order_data['timestamp'] - timing_data['timestamp']).abs() < 0.001
    return flash_order_events

def quote_stuffing_detection(order_data, order_rate_threshold):
    # Detect quote stuffing by analyzing order rates
    order_rate = len(order_data) / (order_data['time'].max() - order_data['time'].min())
    cancellation_rate = order_data['cancellations'].sum() / len(order_data)
    stuffing_detected = (order_rate > order_rate_threshold) & (cancellation_rate > 0.5)
    return stuffing_detected

def arbitrage_strategy_correlation_pairs(trade_data, correlation_threshold):
    # Implement correlation-based arbitrage strategy
    correlations = trade_data.corr()
    correlated_pairs = correlations[(correlations > correlation_threshold) & (correlations < 1)]
    arbitrage_opportunities = []
    for pair in correlated_pairs.index:
        spread = trade_data[pair[0]] - trade_data[pair[1]]
        if spread.abs().mean() > 0.01:
            arbitrage_opportunities.append(pair)
    return arbitrage_opportunities

def liquidity_mining_algorithm(market_data, volume_data):
    # Mine liquidity by providing and taking liquidity strategically
    liquidity_zones = market_data.groupby('price').sum()['volume']
    mining_opportunities = liquidity_zones[liquidity_zones > volume_data.mean()]
    return mining_opportunities

def stealth_arbitrage(hft_order_data, market_impact_minimization):
    # Perform arbitrage while minimizing market footprint
    stealth_orders = []
    for order in hft_order_data:
        if order['impact'] < market_impact_minimization:
            stealth_orders.append(order)
    return stealth_orders

def subsecond_statistical_arbitrage(subsecond_data, arbitrage_opportunities):
    # Perform arbitrage within sub-second intervals
    subsecond_spreads = subsecond_data['ask'] - subsecond_data['bid']
    profitable_opportunities = subsecond_spreads[subsecond_spreads > arbitrage_opportunities.mean()]
    return profitable_opportunities

def order_book_spoofing_detection(order_book_data, abnormal_patterns):
    # Detect and analyze spoofing patterns in the order book
    spoofing_detected = order_book_data['ask'] > abnormal_patterns['ask'] * 1.1
    return spoofing_detected

def cross_exchange_latency_arbitrage(order_flow_data, latency_diff):
    # Exploit latency differences between exchanges
    arbitrage_opportunity = (order_flow_data['price_diff'] / latency_diff) > 0.01
    return arbitrage_opportunity

def flash_crash_detection(market_data, threshold=0.05):
    # Detect flash crashes in the market
    sudden_drop = (market_data['price'].pct_change() < -threshold)
    flash_crash_detected = any(sudden_drop)
    return flash_crash_detected

def smart_order_routing_hft(order_book_data, route_parameters):
    # Implement smart order routing for HFT
    best_routes = []
    for route in route_parameters:
        cost = order_book_data['price'] * route['fee']
        if cost < order_book_data['best_bid']:
            best_routes.append(route)
    return best_routes

def optimal_trade_execution_timing(order_book_data, timing_model):
    # Determine optimal timing for trade execution
    optimal_time = timing_model.predict(order_book_data)
    return optimal_time

# Function to intelligently activate multiple strategies
def intelligent_strategy_activation(order_book_data, market_conditions):
    strategies = {
        'latency_arbitrage': latency_arbitrage,
        'market_microstructure_analysis': market_microstructure_analysis,
        'quote_stuffing_detection': quote_stuffing_detection,
        'momentum_ignition': momentum_ignition,
        'spoofing_detection': spoofing_detection,
        'order_anticipation': order_anticipation,
        'order_book_imbalance': order_book_imbalance,
        'flash_ordering': flash_ordering,
        'passive_aggressive_order_execution': passive_aggressive_order_execution,
        'optimal_liquidity_consumption': optimal_liquidity_consumption,
        'multi_leg_arbitrage': multi_leg_arbitrage,
        'high_frequency_mean_reversion': high_frequency_mean_reversion,
        'colocation_arbitrage': colocation_arbitrage,
        'liquidity_detection': liquidity_detection,
        'adaptive_hft_strategy': adaptive_hft_strategy,
        'market_manipulation_detection': market_manipulation_detection,
        'flash_order_detection': flash_order_detection,
        'quote_stuffing_detection': quote_stuffing_detection,
        'arbitrage_strategy_correlation_pairs': arbitrage_strategy_correlation_pairs,
        'liquidity_mining_algorithm': liquidity_mining_algorithm,
        'stealth_arbitrage': stealth_arbitrage,
        'subsecond_statistical_arbitrage': subsecond_statistical_arbitrage,
        'order_book_spoofing_detection': order_book_spoofing_detection,
        'cross_exchange_latency_arbitrage': cross_exchange_latency_arbitrage,
        'flash_crash_detection': flash_crash_detection,
        'smart_order_routing_hft': smart_order_routing_hft,
        'optimal_trade_execution_timing': optimal_trade_execution_timing
    }
    active_strategies = []
    for strategy_name, strategy_func in strategies.items():
        result = strategy_func(order_book_data, market_conditions)
        if result:
            active_strategies.append(strategy_name)
    return active_strategies

# Function to execute all strategies
def execute_all_strategies(order_book_data, market_conditions):
    strategies = intelligent_strategy_activation(order_book_data, market_conditions)
    results = []
    for strategy in strategies:
        result = globals()[strategy](order_book_data, market_conditions)
        results.append(result)
        if 'expected_return' in result and result['expected_return'] > 100:
            send_alert(result)
    return results

# Function to send alerts for high return stocks
def send_alert(result):
    if result['expected_return'] > 100:
        message = f"High Return Stock Alert! {result['stock']}: Expected Return {result['expected_return']}%"
        send_email_alert(message)
        send_sms_alert(message)

# Function to send email alert
def send_email_alert(message):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = EMAIL_ADDRESS
    msg['Subject'] = "Stock Alert"
    msg.attach(MIMEText(message, 'plain'))
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    text = msg.as_string()
    server.sendmail(EMAIL_ADDRESS, EMAIL_ADDRESS, text)
    server.quit()

# Function to send SMS alert
def send_sms_alert(message):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=TO_PHONE_NUMBER
    )

# Real-time risk management
def real_time_risk_management(results):
    total_exposure = sum([result['exposure'] for result in results if 'exposure' in result])
    max_exposure = 1000000  # Example risk limit
    if total_exposure > max_exposure:
        print("Risk limit breached! Taking corrective actions.")
        # Implement corrective actions (e.g., hedge positions, limit orders)

# Generate CSV file with enhanced details
def generate_csv_report(results):
    columns = ['Stock', 'Signal', 'Expected Return', 'Justification', 'Risk Assessment', 'Volume', 'Market Impact', 'Timestamp', 'SEC Compliance', 'Regulation Remarks']
    data = []
    for result in results:
        if 'stock' in result:
            data.append([
                result.get('stock', ''),
                result.get('signal', ''),
                result.get('expected_return', 0),
                result.get('justification', ''),
                result.get('risk_assessment', ''),
                result.get('volume', 0),
                result.get('market_impact', 0),
                datetime.now(),
                result.get('sec_compliance', True),
                result.get('regulation_remarks', '')
            ])
    df = pd.DataFrame(data, columns=columns)
    csv_filename = f'stock_recommendations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(csv_filename, index=False)
    csv_data = df.values.tolist()
    save_csv_to_db(csv_data)  # Save to database

# Schedule and execute strategies at market open
def schedule_daily_execution():
    market_open_time = "09:30:00"
    create_table()
    while True:
        current_time = time.strftime("%H:%M:%S")
        if current_time == market_open_time:
            order_book_data = {
                'ask': [100, 101, 102], 'bid': [99, 98, 97], 
                'ask_size': [100, 200, 150], 'bid_size': [150, 250, 200]
            }
            market_conditions = {'volatility': 0.5, 'volume': 1000}
            results = execute_all_strategies(order_book_data, market_conditions)
            real_time_risk_management(results)
            generate_csv_report(results)
            break
        time.sleep(1)

if __name__ == "__main__":
    schedule_daily_execution()
