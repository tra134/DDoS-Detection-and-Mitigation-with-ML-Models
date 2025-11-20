#DDoS Detection and Mitigation with ML Models

A comprehensive system for detecting and mitigating DDoS attacks in real-time using Machine Learning models and NS-3 network simulations. This project implements a closed-loop feedback system where NS-3 simulates network traffic, and a Python-based AI agent detects and mitigates attacks on the fly.

📋 Project Overview

This project implements a complete DDoS detection and mitigation pipeline that combines:

    NS-3 Network Simulations: Realistic simulation of IoT networks under DDoS attack (UDP Flood).
    Machine Learning Models: Pre-trained Random Forest models for accurate attack detection.
    Real-time Mitigation System: A closed-loop system where Python reads live network stats and instructs NS-3 to block malicious IPs instantly.
    Performance Analysis: Tools to visualize Latency, Throughput, PDR, and Accuracy across different scenarios.

📂 Project Structure

```bash
ddos-project-new/
├── analysis/ # Scripts for performance visualization
├── config/ # YAML configuration files
├── data/
│ └── raw/ # Offline simulation results
├── live/ # Real-time communication pipe files
├── models/ # Saved ML models (.pkl/.joblib)
├── mitigation/
│ └── mitigator.py # AI mitigation engine
├── ml-pipeline/ # ML training scripts
├── ns3-simulations/
│ └── src/
│ └── ddos-simulator.cc # Main C++ logic
├── run_experiments.sh # Automated multi-scenario runner
├── run_visualization.py # Plot results
└── requirements.txt # Python dependencies
```

🚀 Quick Start

1. Prerequisites
    OS: Linux (Ubuntu recommended)
    Software: NS-3 (version 3.35+), Python 3.8+, GCC, CMake

2. Installation
Clone the repository:
Bash
git clone https://github.com/tra134/DDoS-Detection-and-Mitigation-with-ML-Model
cd ddos-project-new

Set up Python environment:
Bash
python3 -m venv ddos-env
source ddos-env/bin/activate
pip install -r requirements.txt

Build the NS-3 Simulation:
Bash
cd ns3-simulations/build
cmake ..
make -j$(nproc)
cd ../.. 

3. Running the Project (Two Modes)

A. Real-time Mitigation Mode (The "Cool" Stuff)
See the AI detect and block attacks while the simulation runs.

You need two separate terminals.
Terminal 1: The AI Agent (Defender)
Bash
# Activate env
source ddos-env/bin/activate
# Run the mitigator
python3 mitigation/mitigator.py
Wait until you see: "✅ AI Brain is ready. Waiting for data..."
Terminal 2: The Simulation (Network)
Bash

# Go to build directory
cd ns3-simulations/build
# Run simulation (e.g., 30 nodes, 10 attackers)
./ddos-simulator --nodes=30 --attackers=10 --time=60
Watch Terminal 1 detect the attack and Terminal 2 confirm the blocked packets!

B. Performance Analysis Mode (Automated)
Run multiple scenarios to generate charts for Latency, PDR, etc.
Bash

# Make sure you are in the project root
chmod +x run_experiments.sh
./run_experiments.sh

This script will:
    Run simulations for 10, 20, 30, 40, 50 nodes.
    Collect detailed logs.
    Automatically generate performance charts in the results/ folder.

📊 Performance Metrics

The system evaluates network health using four key metrics:
    Latency: Measures network delay. High latency indicates congestion.
    Throughput: Data transfer rate. A drop indicates successful DDoS.
    Packet Delivery Ratio (PDR): Percentage of packets successfully delivered.
    Detection Accuracy: How accurately the ML model identifies attack flows.

Sample Result:
    With mitigation enabled, PDR for legitimate users maintains >60% even under 50Mbps attack load on a 5Mbps link, whereas without mitigation, it would drop to near 0%.

🎯 ML Models Implemented
    Random Forest (Best Performer): Selected for its high accuracy and ability to handle tabular network flow data.
    Gradient Boosting: Used for comparison.
    SVM (Linear): Lightweight alternative.

🔧 Configuration

NS-3 Parameters:
    Topology: Star/P2P mixed with IoT nodes.
    Bottleneck Link: 5Mbps (to simulate realistic congestion).
    Attack Traffic: UDP Flood (5000kbps per attacker).

Feature Selection: The model is trained on Cumulative Flow Statistics:
    tx_packets, tx_bytes (Volume)
    packet_loss_ratio (Impact)
    flow_duration

🛡️ How Mitigation Works

    Monitor: NS-3 calculates cumulative flow statistics every second and writes to data/live/live_flow_stats.csv.
    Detect: Python script (mitigator.py) watches this file, preprocesses data, and feeds it to the Random Forest model.
    Decide: If a flow is predicted as "Attack", the source IP is written to data/live/blacklist.txt.
    Act: NS-3 reads the blacklist and the PacketDropCallback function instantly drops any new packets from those IPs at the Base Station level.


📄 License

This project is licensed under the MIT License.

🏆 Acknowledgments

    NS-3 Network Simulator Project

    Scikit-learn Community
