#!/usr/bin/env python3
import sys
import os

# Add current directory to path
sys.path.append('.')

from analysis.performance_visualization import AccurateNS3Analyzer

def main():
    print("🚀 STEP 6: ACCURATE PERFORMANCE ANALYSIS")
    print("=" * 60)
    print("📊 Generating 4 performance metrics from REAL NS-3 data:")
    print("   6.1: Number of IOT Nodes vs. Latency (ms)")
    print("   6.2: Number of IOT Nodes vs. Throughput (Kbps)") 
    print("   6.3: Number of IOT Nodes vs. Packet Delivery Ratio (%)")
    print("   6.4: Number of IOT Nodes vs. Detection Accuracy (%)")
    print("=" * 60)
    
    # --- CẤU HÌNH ĐƯỜNG DẪN (SỬA LỖI Ở ĐÂY) ---
    # Sử dụng đường dẫn tuyệt đối chính xác từ lệnh pwd của bạn
    base_dir = '/home/traphan/ns-3-dev/ddos-project-new'
    
    data_dir = os.path.join(base_dir, 'data', 'raw')
    model_path = os.path.join(base_dir, 'models', 'ddos_model.pkl')
    node_scenarios = [10, 20, 30, 40, 50]
    
    # Kiểm tra data
    print(f"🔍 Checking for simulation data in: {data_dir}")
    
    if not os.path.exists(data_dir):
         print(f"❌ Directory not found: {data_dir}")
         return

    data_files = [f for f in os.listdir(data_dir) if f.startswith('ns3_detailed_results_')]
    
    if not data_files:
        print("❌ No simulation data found!")
        print("💡 Please run multiple NS-3 simulations first:")
        print("   - 10, 20, 30, 40, 50 IoT nodes")
        print("   - Results should be saved as: ns3_detailed_results_XX_nodes.csv")
        return
    
    print(f"✅ Found {len(data_files)} simulation files")
    
    # Chạy analysis
    analyzer = AccurateNS3Analyzer(model_path=model_path)
    metrics_df = analyzer.analyze_scenario_files(data_dir, node_scenarios)
    
    if not metrics_df.empty:
        analyzer.plot_accurate_metrics(metrics_df)
        print("\n🎉 STEP 6 COMPLETED SUCCESSFULLY!")
        print("📈 All 4 performance metrics generated from REAL data")
    else:
        print("\n❌ Analysis failed - no valid data processed")

if __name__ == "__main__":
    main()