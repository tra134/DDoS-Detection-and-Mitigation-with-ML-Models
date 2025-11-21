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
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


class NS3FlowAnalyzer:
    """Analyzer tự động đọc nhiều file FlowMonitor CSV từ NS-3"""

    def __init__(self):
        self.df_all = pd.DataFrame()
        self.metrics_df = pd.DataFrame()

    def load_all_csv(self, pattern=None):
        """Load tất cả file CSV theo pattern"""
        if pattern is None:
            pattern = os.path.join(RESULTS_DIR, "ns3_detailed_results_*.csv")

        files = glob.glob(pattern)
        if not files:
            print(" Do not see the CSV file.")
            return False

        df_list = []
        for f in files:
            try:
                df = pd.read_csv(f, skipinitialspace=True)
                df.columns = df.columns.str.strip()
                df_list.append(df)
                print(f"Loaded {f} ({df.shape[0]} rows)")
            except Exception as e:
                print(f"Error reading {f}: {e}")

        if df_list:
            self.df_all = pd.concat(df_list, ignore_index=True)
            self.df_all.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.df_all.fillna(0, inplace=True)
            print(f"Total rows: {self.df_all.shape[0]}")
            return True

        return False

    def calculate_metrics(self, scenario_col=None):
        """Tính metrics cho mỗi scenario"""
        if self.df_all.empty:
            print("No data loaded.")
            return

        # Nếu không có cột scenario -> tạo scenario tự động
        if scenario_col is None:
            self.df_all['scenario'] = np.arange(len(self.df_all))
        else:
            self.df_all['scenario'] = self.df_all[scenario_col]

        metrics = []
        for scenario, df_s in self.df_all.groupby('scenario'):

            benign = df_s[df_s['label'] == 0]
            latency = 0.0
            throughput = 0.0
            pdr = 0.0

            if not benign.empty:

                # --- Latency: delay_sum đơn vị = giây -> đổi sang ms ---
                valid_rx = benign[benign['rx_packets'] > 0]
                if not valid_rx.empty:
                    latency = (valid_rx['delay_sum'] / valid_rx['rx_packets']).mean() * 1000  # <-- FIXED: *1000

                # Throughput
                valid_thr = benign[benign['throughput'] > 0]
                if not valid_thr.empty:
                    throughput = valid_thr['throughput'].mean()

                # Packet Delivery Ratio
                pdr = 1 - benign['packet_loss_ratio'].mean()

            metrics.append({
                'scenario': scenario,
                'latency_ms': latency,
                'throughput_kbps': throughput,
                'packet_delivery_ratio': pdr
            })

        self.metrics_df = pd.DataFrame(metrics)
        return self.metrics_df

    def plot_metrics(self, save_path=None):
        """Vẽ biểu đồ tổng quan metrics"""
        if self.metrics_df.empty:
            print(" No metrics calculated.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # Latency
        axes[0].plot(self.metrics_df['scenario'], self.metrics_df['latency_ms'],
                     marker='o', color='red', linewidth=2)
        axes[0].set_title('Latency per Scenario (ms)')
        axes[0].set_xlabel('Scenario')
        axes[0].set_ylabel('Latency (ms)')
        axes[0].grid(True, alpha=0.3)

        # Throughput
        axes[1].plot(self.metrics_df['scenario'], self.metrics_df['throughput_kbps'],
                     marker='s', color='green', linewidth=2)
        axes[1].set_title('Throughput per Scenario (Kbps)')
        axes[1].set_xlabel('Scenario')
        axes[1].set_ylabel('Throughput (Kbps)')
        axes[1].grid(True, alpha=0.3)

        # PDR
        axes[2].plot(self.metrics_df['scenario'], self.metrics_df['packet_delivery_ratio'] * 100,
                     marker='^', color='blue', linewidth=2)
        axes[2].set_title('Packet Delivery Ratio per Scenario (%)')
        axes[2].set_xlabel('Scenario')
        axes[2].set_ylabel('PDR (%)')
        axes[2].set_ylim(0, 100)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()


# Example usage:
# analyzer = NS3FlowAnalyzer()
# analyzer.load_all_csv()
# analyzer.calculate_metrics()
# analyzer.plot_metrics(save_path=os.path.join(RESULTS_DIR, 'ns3_metrics.png'))
