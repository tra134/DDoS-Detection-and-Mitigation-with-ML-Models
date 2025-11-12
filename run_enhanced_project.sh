#!/bin/bash

# Enhanced DDoS Detection Project Runner
echo "=== Enhanced DDoS Detection Project ==="

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

# Create necessary directories
mkdir -p results/plots results/models results/data

# Run the enhanced pipeline
echo "Starting enhanced DDoS detection pipeline..."
python ml-pipeline/main.py --config config/experiment.yaml

# Deactivate virtual environment
deactivate
echo "=== Enhanced project execution completed ==="