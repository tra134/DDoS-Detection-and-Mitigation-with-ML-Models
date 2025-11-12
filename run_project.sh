#!/bin/bash

# DDoS Detection Project Runner
echo "=== DDoS Detection Project ==="

# Check if virtual environment exists
if [ ! -d "ddos-env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv ddos-env
fi

# Activate virtual environment
echo "Activating virtual environment..."
source ddos-env/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the main pipeline
echo "Starting DDoS detection training..."
python ml-pipeline/main.py --config config/experiment.yaml

# Deactivate virtual environment
deactivate
echo "=== Project execution completed ==="