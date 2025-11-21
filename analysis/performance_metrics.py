import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import warnings
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 12

# --- CẤU HÌNH ĐƯỜNG DẪN ---
PROJECT_ROOT = '/home/traphan/ns-3-dev/ddos-project-new'
DATA_DIR = os.path.join(PROJECT_ROOT, 'data/raw')      # thư mục CSV
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')    # thư mục lưu plot
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class NS3FlowAnalyzer:
    """Analyzer tự động đọc nhiều file FlowMonitor CSV từ NS-3"""
    
    def __init__(self):
        self.df_all = pd.DataFrame()
        self.metrics_df = pd.DataFrame()
    
    def load_all_csv(self, pattern=None):
        """Load tất cả file CSV theo pattern"""
        if pattern is None:
            pattern = os.path.join(DATA_DIR, "ns3_detailed_results_*.csv")
        files = glob.glob(pattern)
        if not files:
            print(" Don't find CSV files...")
            return False
        
        df_list = []
        for f in files:
            try:
                df = pd.read_csv(f, skipinitialspace=True)
                df.columns = df.columns.str.strip()
                df_list.append(df)
                print(f"Loaded {f} ({df.shape[0]} rows)")
            except Exception as e:
                print(f" Error reading {f}: {e}")
        
        if df_list:
            self.df_all = pd.concat(df_list, ignore_index=True)
            # fill nan/infinity
            self.df_all.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.df_all.fillna(0, inplace=True)
            print(f"Total rows: {self.df_all.shape[0]}")
            return True
        return False
    
    def calculate_metrics(self, scenario_col=None):
        """
        Tính metrics cho mỗi scenario, theo Hướng 1:
        → Dùng TẤT CẢ các flow có rx_packets > 0, không lọc benign nữa.
        """
        if self.df_all.empty:
            print("No data loaded.")
            return
        
        # Nếu không có cột scenario, tự tạo 'scenario'
        if scenario_col is None:
            self.df_all['scenario'] = np.arange(len(self.df_all))
        else:
            self.df_all['scenario'] = self.df_all[scenario_col]
        
        metrics = []
        for scenario, df_s in self.df_all.groupby('scenario'):
            valid = df_s[df_s['rx_packets'] > 0]

            latency = 0.0
            throughput = 0.0
            pdr = 0.0
            
            if not valid.empty:
                latency = (valid['delay_sum'] / valid['rx_packets']).mean() * 1000
                
                valid_tp = valid[valid['throughput'] > 0]
                if not valid_tp.empty:
                    throughput = valid_tp['throughput'].mean()
                
                pdr = 1 - valid['packet_loss_ratio'].mean()
            
            metrics.append({
                'scenario': scenario,
                'latency_ms': latency,
                'throughput_kbps': throughput,
                'packet_delivery_ratio': pdr
            })
        
        self.metrics_df = pd.DataFrame(metrics)
        return self.metrics_df
    
    def plot_metrics(self, save_path=None, metrics_to_plot=None):
        """
        Vẽ biểu đồ cho tất cả metrics.

        - metrics_to_plot: list tên cột trong self.metrics_df để vẽ. 
          Nếu None → tự động lấy tất cả trừ cột 'scenario'.
        """
        if self.metrics_df.empty:
            print(" No metrics calculated.")
            return
        
        if metrics_to_plot is None:
            metrics_to_plot = [c for c in self.metrics_df.columns if c != 'scenario']
        
        n = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
        
        if n == 1:
            axes = [axes]
        
        for i, col in enumerate(metrics_to_plot):
            y = self.metrics_df[col]
            
            # Nếu giá trị 0-1, hiển thị %
            if y.max() <= 1:
                y_plot = y*100
                ylabel = f"{col} (%)"
            else:
                y_plot = y
                ylabel = col
            
            axes[i].plot(self.metrics_df['scenario'], y_plot, marker='o', linewidth=2)
            axes[i].set_title(f"{col} per Scenario")
            axes[i].set_xlabel("Scenario")
            axes[i].set_ylabel(ylabel)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, max(y_plot.max()*1.1, 1))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Plot saved to {save_path}")
        plt.show()

# --------------------------
if __name__ == "__main__":
    analyzer = NS3FlowAnalyzer()
    if analyzer.load_all_csv():
        analyzer.calculate_metrics()
        plot_file = os.path.join(RESULTS_DIR, 'ns3_all_metrics.png')
        analyzer.plot_metrics(save_path=plot_file)
