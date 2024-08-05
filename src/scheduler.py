# src/scheduler.py
import schedule
import time
from trade_execution import execute_trades
from data_acquisition import fetch_stock_data
from trading_strategies import apply_strategies
from csv_generation import generate_csv_report
from apscheduler.schedulers.blocking import BlockingScheduler
import logging
import smtplib
from email.mime.text import MIMEText
from twilio.rest import Client
import requests
from datetime import datetime
import pandas as pd
from src.sentiment_analysis import analyze_market_sentiment
from src.smart_order_routing import optimal_order_routing

logging.basicConfig(level=logging.INFO)

# Notification configuration
EMAIL_ADDRESS = "shrqshhb@gmail.com"
PHONE_NUMBER = "+16465587623"
TWILIO_ACCOUNT_SID = "your_twilio_account_sid"
TWILIO_AUTH_TOKEN = "your_twilio_auth_token"
TWILIO_PHONE_NUMBER = "your_twilio_phone_number"
ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_api_key"

def send_email(subject, message):
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = EMAIL_ADDRESS

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(EMAIL_ADDRESS, 'your_email_password')
        server.sendmail(EMAIL_ADDRESS, EMAIL_ADDRESS, msg.as_string())

def send_text(message):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=PHONE_NUMBER
    )

def notify_big_shift_and_recommendations(shift_info, recommendations):
    # Notify user about big shifts and recommendations
    subject = "Stock Alert: Significant Market Shift Detected"
    message = f"Market Shift: {shift_info}\nRecommendations: {recommendations}"
    send_email(subject, message)
    send_text(message)

def detect_big_shift_and_notify():
    # Dummy function to detect big market shifts
    shift_info = "Big drop in tech stocks due to earnings reports"
    recommendations = "Short TSLA, Buy AAPL"
    notify_big_shift_and_recommendations(shift_info, recommendations)

def fetch_market_news():
    # Fetch latest market news
    news_api_url = f"https://newsapi.org/v2/everything?q=stock&apiKey=your_news_api_key"
    response = requests.get(news_api_url)
    news_data = response.json()
    return news_data['articles']

def notify_market_news():
    news_articles = fetch_market_news()
    for article in news_articles[:5]:  # Notify about top 5 news
        title = article['title']
        description = article['description']
        url = article['url']
        message = f"News: {title}\nDescription: {description}\nURL: {url}"
        send_email("Market News Alert", message)
        send_text(message)

def daily_trading_routine():
    stock_data = fetch_stock_data()
    signals = apply_strategies(stock_data)
    execute_trades(signals)
    detect_big_shift_and_notify()  # Check and notify about market shifts
    notify_market_news()  # Send market news alerts

def schedule_tasks():
    # Schedule daily tasks
    schedule.every().day.at("09:30").do(daily_trading_routine)
    schedule.every().day.at("09:45").do(generate_csv_report)
    schedule.every().day.at("08:00").do(notify_market_news)  # Pre-market news
    schedule.every().day.at("15:45").do(generate_csv_report)  # Post-market report

    while True:
        schedule.run_pending()
        time.sleep(1)

def schedule_event_based_tasks(event_list, event_action):
    for event in event_list:
        schedule.every().day.at(event['time']).do(event_action, event['details'])

def schedule_real_time_monitoring(api_keys, interval=60):
    def real_time_monitor():
        # Placeholder for real-time monitoring logic
        print("Real-time monitoring active...")

    schedule.every(interval).seconds.do(real_time_monitor)

def schedule_daily_tasks(tasks, start_hour, start_minute):
    for task in tasks:
        schedule.every().day.at(f"{start_hour}:{start_minute}").do(task)

def schedule_weekly_tasks(tasks, day_of_week, hour, minute):
    for task in tasks:
        schedule.every().week.do(task).at(f"{hour}:{minute}").do(task)

def schedule_monthly_tasks(tasks, day_of_month, hour, minute):
    for task in tasks:
        schedule.every().month.at(f"{day_of_month} {hour}:{minute}").do(task)

def schedule_real_time_tasks(tasks, interval_seconds):
    for task in tasks:
        schedule.every(interval_seconds).seconds.do(task)

def custom_holiday_schedule(tasks, holidays):
    for holiday in holidays:
        for task in tasks:
            schedule.every().day.at(holiday).do(task)

def intraday_scheduling(tasks, intervals):
    for interval in intervals:
        for task in tasks:
            schedule.every(interval).minutes.do(task)

def conditional_scheduling(tasks, conditions):
    for task, condition in zip(tasks, conditions):
        if condition():
            schedule.every().day.do(task)

def adaptive_scheduling(tasks, market_volatility):
    for task in tasks:
        if market_volatility > threshold:
            schedule.every().day.do(task)

def multi_market_scheduling(tasks, market_hours):
    for market in market_hours:
        for task in tasks:
            schedule.every().day.at(market).do(task)

def predictive_scheduling(tasks, market_predictors):
    for predictor in market_predictors:
        if predictor():
            for task in tasks:
                schedule.every().day.do(task)

def algorithmic_scheduling(algorithms, optimal_times):
    for algo, time in zip(algorithms, optimal_times):
        schedule.every().day.at(time).do(algo)

def load_balancing_scheduling(tasks, system_load):
    if system_load < threshold:
        for task in tasks:
            schedule.every().day.do(task)

def market_open_scheduling(tasks, market_open_time):
    for task in tasks:
        schedule.every().day.at(market_open_time).do(task)

def economic_event_scheduling(tasks, economic_calendar):
    for event in economic_calendar:
        for task in tasks:
            schedule.every().day.at(event).do(task)

def news_event_scheduling(tasks, news_alerts):
    for alert in news_alerts:
        for task in tasks:
            schedule.every().day.at(alert).do(task)

def predictive_task_prioritization(tasks, market_forecasts):
    for forecast in market_forecasts:
        if forecast():
            for task in tasks:
                schedule.every().day.do(task)

def global_market_scheduling(tasks, global_times):
    for time in global_times:
        for task in tasks:
            schedule.every().day.at(time).do(task)

def volatility_based_scheduling(tasks, volatility_indices):
    for index in volatility_indices:
        if index > threshold:
            for task in tasks:
                schedule.every().day.do(task)

def event_based_stock_monitoring(stock_events):
    for event in stock_events:
        schedule.every().day.at(event['time']).do(notify_big_shift_and_recommendations, event['shift_info'], event['recommendations'])

def sentiment_and_volatility_analysis():
    # Perform sentiment analysis and correlate with market volatility
    sentiment_scores = analyze_market_sentiment()
    market_volatility = np.random.random()
    if sentiment_scores > 0.7 and market_volatility > 0.5:
        notify_big_shift_and_recommendations("High sentiment and volatility detected", "Consider buying growth stocks")

def optimal_order_routing_scheduling():
    # Implement optimal order routing based on current market conditions
    schedule.every().day.at("09:35").do(optimal_order_routing, "Order book data", 100, "High")

# New Features

def crypto_market_news_analysis():
    # Fetch and analyze crypto market news
    crypto_news_api_url = f"https://newsapi.org/v2/everything?q=crypto&apiKey=your_news_api_key"
    response = requests.get(crypto_news_api_url)
    news_data = response.json()
    for article in news_data['articles'][:5]:
        title = article['title']
        description = article['description']
        url = article['url']
        message = f"Crypto News: {title}\nDescription: {description}\nURL: {url}"
        send_email("Crypto Market News Alert", message)
        send_text(message)

def schedule_crypto_tasks():
    # Schedule tasks specific to crypto trading
    schedule.every().day.at("07:00").do(crypto_market_news_analysis)
    schedule.every().day.at("17:00").do(generate_csv_report)

def economic_indicator_monitoring():
    # Monitor economic indicators and adjust strategies
    economic_indicators = {
        'GDP': 2.5,
        'Unemployment': 4.0,
        'CPI': 1.8
    }
    if economic_indicators['GDP'] < 2.0:
        send_text("Economic Alert: GDP growth slowing, adjust strategy.")
    if economic_indicators['Unemployment'] > 5.0:
        send_email("Economic Alert", "High unemployment rate detected, consider defensive positions.")

def technical_indicators_monitoring():
    # Monitor technical indicators and send alerts
    rsi_data = {'AAPL': 70, 'GOOGL': 30}
    for stock, rsi in rsi_data.items():
        if rsi > 70:
            send_text(f"Overbought Alert: {stock} RSI is {rsi}. Consider selling.")
        elif rsi < 30:
            send_email("Oversold Alert", f"{stock} RSI is {rsi}. Consider buying.")

# Main
if __name__ == "__main__":
    schedule_tasks()
    schedule_crypto_tasks()
    economic_indicator_monitoring()
    technical_indicators_monitoring()
    sentiment_and_volatility_analysis()
    optimal_order_routing_scheduling()
    event_based_stock_monitoring([{'time': '09:45', 'shift_info': 'Tech stocks', 'recommendations': 'Buy AAPL'}])
