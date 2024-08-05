# src/backtesting.py
import pandas as pd
import numpy as np

def backtest_strategy(data, strategy, initial_cash=100000, leverage=2, transaction_cost=0.001):
    cash = initial_cash
    holdings = 0
    total_value_history = []
    drawdowns = []
    alpha = 0  # For calculating Jensen's alpha

    for i in range(1, len(data)):
        signal = strategy(data.iloc[i - 1:i])
        if signal == 'buy':
            holdings = cash / data.iloc[i]['Close'] * leverage
            cash = 0
        elif signal == 'sell':
            cash = holdings * data.iloc[i]['Close'] * (1 - transaction_cost)
            holdings = 0
        total_value = cash + holdings * data.iloc[-1]['Close']
        total_value_history.append(total_value)
        drawdown = (max(total_value_history) - total_value) / max(total_value_history)
        drawdowns.append(drawdown)
        alpha += data.iloc[i]['excess_return'] - (data.iloc[i]['beta'] * data.iloc[i]['market_return'])

    return total_value, total_value_history, drawdowns, alpha

def calculate_performance(total_value, initial_cash=100000, drawdowns=None):
    return_rate = (total_value - initial_cash) / initial_cash
    sharpe_ratio = return_rate / np.std(return_rate)
    max_drawdown = max(drawdowns) if drawdowns else 0
    calmar_ratio = return_rate / max_drawdown if max_drawdown != 0 else float('inf')
    jensen_alpha = total_value - initial_cash - max(drawdowns)
    return return_rate, sharpe_ratio, max_drawdown, calmar_ratio, jensen_alpha

def monte_carlo_simulation(data, num_simulations=10000, time_horizon=252):
    # Monte Carlo simulation for risk assessment and scenario analysis
    results = []
    for _ in range(num_simulations):
        simulated_path = [data.iloc[0]['Close']]
        for t in range(1, time_horizon):
            next_price = simulated_path[-1] * np.exp(np.random.normal(0, 1))
            simulated_path.append(next_price)
        results.append(simulated_path)
    return np.array(results)

def scenario_analysis(data, scenarios):
    # Scenario analysis for stress testing
    scenario_results = []
    for scenario in scenarios:
        modified_data = data.copy()
        modified_data['Close'] *= scenario['price_change']
        total_value, _, _, _ = backtest_strategy(modified_data, scenario['strategy'])
        scenario_results.append(total_value)
    return scenario_results
