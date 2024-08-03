import os

# Define the directory structure and files
structure = {
    "trading_system": {
        "data": {},
        "src": {
            "data_acquisition.py": "",
            "mathematical_techniques.py": "",
            "linear_algebra_techniques.py": "",
            "statistical_techniques.py": "",
            "trading_strategies.py": "",
            "advanced_methods.py": "",
            "execution.py": "",
            "scheduler.py": "",
        },
        "tests": {},
        "README.md": "# Advanced Trading System\n\nThis is a README file.",
        "LICENSE": "MIT License\n\nPermission is hereby granted, free of charge, to any person obtaining a copy...",
        ".gitignore": "# Python\n*.pyc\n__pycache__/\n\n# Environment\n.env\n.venv\nenv/\nvenv/\nENV/\n\n# Jupyter Notebooks\n.ipynb_checkpoints/\n\n# Logs\nlogs/\n*.log\n\n# Data\ndata/",
    }
}

# Function to create the directory structure
def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            # Create directory
            os.makedirs(path, exist_ok=True)
            # Recursively create subdirectories and files
            create_structure(path, content)
        else:
            # Create file with content
            with open(path, "w") as file:
                file.write(content)

# Base path where the structure will be created
base_path = "."

# Create the structure
create_structure(base_path, structure)

print("Directory structure and files created successfully!")

