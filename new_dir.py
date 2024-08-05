# new_dir_structure_v5.py
import os

structure = {
    "data": ["raw", "processed", "financial_statements", "logs", "alternative_data", "crypto_data", "daily_reports"],
    "src": ["__init__.py", "data_acquisition.py", "data_processing.py", "fundamental_analysis.py", "technical_analysis.py", 
            "trading_strategies.py", "mathematical_models.py", "machine_learning.py", "risk_management.py", 
            "trade_execution.py", "backtesting.py", "utils.py", "scheduler.py", "blockchain_analytics.py", 
            "regtech_compliance.py", "smart_order_routing.py", "hft_strategies.py", "advanced_machine_learning.py",
            "order_execution.py", "sentiment_analysis.py", "portfolio_management.py", "crypto_analysis.py", 
            "advanced_quant.py", "advanced_mathematics.py", "advanced_analytics.py", "web_traffic_analysis.py", 
            "satellite_imagery_analysis.py", "alternative_data_processing.py", "regulatory_monitoring.py"],
    "tests": ["__init__.py", "test_data_acquisition.py", "test_trading_strategies.py", "test_trade_execution.py", 
              "test_machine_learning.py", "test_utils.py"],
    "scripts": ["deploy.sh", "setup_environment.sh"],
    "config": ["config.yaml", "credentials.json", "logging_config.json"],
    "notebooks": ["exploratory_data_analysis.ipynb", "model_training.ipynb", "strategy_backtesting.ipynb"],
    "root_files": ["requirements.txt", ".gitignore", "README.md", "LICENSE"]
}

def create_structure():
    for dir_name, files in structure.items():
        if dir_name != "root_files":
            dir_path = os.path.join(dir_name)
            os.makedirs(dir_path, exist_ok=True)
            for file in files:
                file_path = os.path.join(dir_path, file)
                open(file_path, 'w').close()
        else:
            for file in files:
                open(file, 'w').close()

    print("New directory structure created successfully.")

if __name__ == "__main__":
    create_structure()

