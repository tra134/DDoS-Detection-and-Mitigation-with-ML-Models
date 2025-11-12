#!/bin/bash

echo "🚀 Starting NS-3 DDoS Simulation..."

# Check if NS-3 environment is set up
if [ -z "$NS3_HOME" ]; then
    echo "❌ NS3_HOME is not set. Please set NS3_HOME environment variable."
    exit 1
fi

cd $NS3_HOME

# Build the simulation
echo "📦 Building DDoS simulation..."
./waf --build ddos-simulator

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

# Create output directory
OUTPUT_DIR="ddos-project-new/ns3-simulations/results/$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Run simulation with parameters
echo "🎯 Running simulation..."
./waf --run "ddos-simulator \
    --nodes=20 \
    --attackers=5 \
    --time=60.0 \
    --netanim=true" \

# Move output files to results directory
mv ddos-animation.xml $OUTPUT_DIR/
mv realtime_stats.csv $OUTPUT_DIR/
mv ns3_detailed_results.csv $OUTPUT_DIR/
mv ddos_data.dat $OUTPUT_DIR/

echo "✅ Simulation completed!"
echo "📁 Results saved to: $OUTPUT_DIR"
echo ""
echo "📊 Generated files:"
echo "   - Animation: $OUTPUT_DIR/ddos-animation.xml"
echo "   - Real-time stats: $OUTPUT_DIR/realtime_stats.csv"
echo "   - Detailed results: $OUTPUT_DIR/ns3_detailed_results.csv"
echo "   - Gnuplot data: $OUTPUT_DIR/ddos_data.dat"