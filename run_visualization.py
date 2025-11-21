#!/usr/bin/env python3
import sys
import os

# Add current directory to path
sys.path.append('.')

from analysis.performance_visualization import AccurateNS3Analyzer

def main():
    print(" STEP 6: ACCURATE PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # --- CẤU HÌNH ĐƯỜNG DẪN TUYỆT ĐỐI ---
    BASE_DIR = '/home/traphan/ns-3-dev/ddos-project-new'
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'ddos_model.pkl')
    node_scenarios = [10, 20, 30, 40, 50]
    
    if not os.path.exists(DATA_DIR):
         print(f" Directory not found: {DATA_DIR}")
         return

 
    analyzer = AccurateNS3Analyzer(model_path=MODEL_PATH)
    metrics_df = analyzer.analyze_scenario_files(DATA_DIR, node_scenarios)
    
    if not metrics_df.empty:
        analyzer.plot_accurate_metrics(metrics_df)
        print("\nALL PLOTS SAVED SUCCESSFULLY TO: /home/traphan/ns-3-dev/ddos-project-new/results/")
    else:
        print("\n Analysis failed - no valid data processed")

if __name__ == "__main__":
    main()