#!/bin/bash

echo "🔬 Starting WOA-SSA Optimization and Performance Evaluation..."

# Check if required directories exist
mkdir -p ../results
mkdir -p ../models

# Activate environment (if using virtual environment)
if [ -d "ddos-env" ]; then
    source ddos-env/bin/activate
fi

# Run the comprehensive evaluation
cd ml-pipeline
python advanced_evaluation.py

echo "✅ WOA-SSA optimization and evaluation completed!"
echo "📊 Results saved to: ../results/"
echo "🤖 Optimized model saved to: ../models/"