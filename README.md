# DDoS-Detection-and-Mitigation-with-ML-Models
A comprehensive system for detecting and mitigating DDoS attacks using Machine Learning models and NS-3 network simulations.
📋 Project Overview

This project implements a complete DDoS detection and mitigation pipeline that combines:

    NS-3 Network Simulations for realistic attack scenarios
    -Machine Learning Models for accurate attack detection
    -Real-time Detection System with automatic mitigation
    -Performance Analysis with multiple metrics visualization


🚀 Quick Start

Installation
Clone the repository
git clone <repository-url>
cd ddos-project-new

Set up Python environment

bash
python -m venv ddos-env
source ddos-env/bin/activate  # On Windows: ddos-env\Scripts\activate
pip install -r requirements.txt

Set up NS-3 environment

bash

cd ns-3-dev
./waf configure

Running the Complete Pipeline

    Train ML Models

bash

python train_quick.py
# or
cd ml-pipeline
python train_model.py

    Run NS-3 Simulations (Multiple Scenarios)

bash

cd ns-3-dev
chmod +x ns3-simulations/run_multiple_scenarios.sh
./ns3-simulations/run_multiple_scenarios.sh

    Run Performance Analysis

bash

python run_accurate_analysis.py

    Start Real-time Detection (Optional)

bash

cd ml-pipeline
python ddos_detector.py

📊 Performance Metrics

The system generates four key performance metrics:

    -IoT Nodes vs. Latency - Network delay under different loads
    -IoT Nodes vs. Throughput - Data transfer capacity
    -IoT Nodes vs. Packet Delivery Ratio - Network reliability
    -IoT Nodes vs. Detection Accuracy - ML model performance

🎯 ML Models Implemented
    Random Forest - Primary detection model
    Decision Tree - Lightweight alternative
    K-Neighbors - Distance-based detection
    Gradient Boosting - Ensemble method
    SVM - Support Vector Machine

WOA-SSA Hybrid Optimization

The system uses Whale Optimization Algorithm (WOA) combined with Squirrel Search Algorithm (SSA) for:
    Feature selection optimization
    Hyperparameter tuning
    Model performance enhancement

🔧 Configuration
NS-3 Simulation Parameters
yaml

# config/ns3-config.yaml
simulation:
  iot_nodes: [10, 20, 30, 40, 50]
  attackers: [2, 4, 6, 8, 10] 
  simulation_time: 60.0
  wifi_standard: "802.11n"

ML Model Configuration
yaml

# config/ml-config.yaml
model:
  threshold: 0.85
  features:
    - packet_length
    - protocol_type
    - packet_rate
    - port_entropy



🛡️ Detection & Mitigation Features
Real-time Detection

    Packet sniffing and analysis

    Statistical pattern recognition

    ML-based classification

    Adaptive thresholding

Mitigation Actions

    Automatic IP blocking

    Rate limiting

    Traffic filtering

    Alert system integration

📈 Results Interpretation
Expected Trends

    Latency: Increases with more IoT nodes

    Throughput: Decreases with network congestion

    PDR: Drops under attack conditions

    Accuracy: Varies based on attack complexity

Performance Benchmarks
    Detection Accuracy: >85%
    False Positive Rate: <5%
    Processing Delay: <10ms per packet
    Mitigation Response: <2 seconds

🔬 Advanced Features
Multi-Scenario Analysis
The system automatically analyzes multiple simulation scenarios:
    Different network sizes (10-50 IoT nodes)
    Various attacker ratios
    Multiple traffic patterns

Optimization Algorithms
    WOA (Whale Optimization): Global search capability
    SSA (Squirrel Search): Local optimization
    Hybrid WOA-SSA: Balanced exploration-exploitation

🐛 Troubleshooting
Common Issues
    NS-3 Build Errors
        Ensure NS-3 dependencies are installed
        Check compiler compatibility

    ML Model Training Failures
        Verify data file paths
        Check feature column names in CSV files

    Visualization Errors
        Ensure matplotlib backend is properly configured
        Check file permissions in results directory




📚 References
    NS-3 Network Simulator Documentation
    Scikit-learn Machine Learning Library
    Whale Optimization Algorithm Research Papers
    DDoS Detection Survey Papers


📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

🏆 Acknowledgments
    NS-3 development team
    Scikit-learn contributors
    Research papers on DDoS detection
    Optimization algorithm researchers

Note: This project is for research and educational purposes. Always ensure proper authorization before deploying in production environments.