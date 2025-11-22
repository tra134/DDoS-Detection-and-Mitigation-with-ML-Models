#!/usr/bin/env python3
"""
accurate_ns3_analyzer_v9.py

- Tính PDR, Latency, Throughput theo node và theo thời gian
- Tự động xác định cột thời gian
- Tích hợp ML model (nếu có)
- Plot metrics có annotation
"""

import os
import glob
import re
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
sns.set_theme()

PROJECT_ROOT = '/home/traphan/ns-3-dev/ddos-project-new'
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

CSV_PATTERN = os.path.join(PROJECT_ROOT, 'data', 'raw', 'ns3_detailed_results*.csv')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'ddos_model.pkl')

class AccurateNS3AnalyzerV9:
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.all_feature_columns = []
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, path):
        try:
            data = joblib.load(path)
            self.model = data.get('model', None)
            self.scaler = data.get('scaler', None)
            self.feature_columns = list(data.get('feature_names', []))
            self.all_feature_columns = list(data.get('all_feature_names', self.feature_columns))
            print("ML model loaded")
        except Exception as e:
            print(f"Failed to load model: {e}")

    def _extract_node_count(self, filename):
        match = re.search(r'ns3_detailed_results_(\d+)', filename)
        return int(match.group(1)) if match else 0

    def _find_time_column(self, df):
        for col in ['time_s','Time','time','simulation_time']:
            if col in df.columns:
                return col
        raise ValueError("Không tìm thấy cột thời gian hợp lệ trong CSV")

    def _calc_metrics_grouped(self, df_group):
        # Tính metrics trung bình cho node
        latency = 0.0
        throughput = 0.0
        pdr = 0.0
        accuracy = 0.0

        if df_group.empty:
            return latency, throughput, pdr, accuracy

        # Latency: trung bình delay_sum / rx_packets
        valid = df_group[df_group['rx_packets']>0]
        if not valid.empty:
            latency = float(np.nanmean(valid['delay_sum'].astype(float)/valid['rx_packets'].astype(float)))

        # Throughput: chuẩn Kbps
        if 'throughput' in df_group.columns:
            arr = df_group['throughput'].fillna(0).astype(float)
            # Nếu throughput quá lớn, chuyển sang Kbps
            if arr.max()>1e6 or np.median(arr)>20000: 
                arr = arr/1000.0
            throughput = float(np.mean(arr))

        # Packet Delivery Ratio: chỉ tính cho lưu lượng benign (label=0)
        if 'label' in df_group.columns:
            benign = df_group[df_group['label']==0]
        else:
            benign = df_group
        if not benign.empty and 'tx_packets' in benign.columns and 'rx_packets' in benign.columns:
            tx_sum = benign['tx_packets'].sum()
            rx_sum = benign['rx_packets'].sum()
            if tx_sum>0: pdr = float(rx_sum/tx_sum)

        # Accuracy nếu có model
        if self.model and 'label' in df_group.columns:
            X_full = df_group.copy()
            for col in self.all_feature_columns:
                if col not in X_full.columns: X_full[col] = 0.0
            X_full = X_full[self.all_feature_columns]
            X_scaled = self.scaler.transform(X_full) if self.scaler else X_full.values
            X_final = pd.DataFrame(X_scaled, columns=self.all_feature_columns)[self.feature_columns]
            y_true = df_group['label'].astype(int)
            y_pred = self.model.predict(X_final)
            accuracy = float(accuracy_score(y_true, y_pred))

        return latency, throughput, pdr, accuracy

    def analyze_all_files(self, csv_pattern=CSV_PATTERN):
        files = glob.glob(csv_pattern)
        if not files:
            print(f"No CSV files found at {csv_pattern}")
            return pd.DataFrame()

        all_data = []
        for f in files:
            node = self._extract_node_count(f)
            df = pd.read_csv(f)
            if df.empty: 
                continue
            df['iot_nodes'] = node
            all_data.append(df)

        df_all = pd.concat(all_data, ignore_index=True)

        metrics_list = []
        for node, group in df_all.groupby('iot_nodes'):
            latency, throughput, pdr, accuracy = self._calc_metrics_grouped(group)
            metrics_list.append({
                'iot_nodes': node,
                'latency_s': latency,
                'throughput_kbps': throughput,
                'packet_delivery_ratio': pdr,
                'accuracy': accuracy
            })

        df_metrics = pd.DataFrame(metrics_list).sort_values('iot_nodes')
        return df_metrics

    def calc_time_series_metrics(self, df):
        # Tính metrics theo thời gian (time series)
        time_col = self._find_time_column(df)
        df_metrics = []
        for t in sorted(df[time_col].unique()):
            df_t = df[df[time_col]==t]
            benign = df_t[df_t.get('label',0)==0]
            tx_sum = benign['tx_packets'].sum() if 'tx_packets' in benign.columns else 0
            rx_sum = benign['rx_packets'].sum() if 'rx_packets' in benign.columns else 0
            pdr = rx_sum/tx_sum if tx_sum>0 else np.nan

            throughput = df_t['throughput'].sum()/1000.0 if 'throughput' in df_t.columns else 0.0
            valid = df_t[df_t['rx_packets']>0]
            latency = float(np.nanmean(valid['delay_sum']/valid['rx_packets'])) if not valid.empty else np.nan

            df_metrics.append({
                'time_s': t,
                'pdr': pdr,
                'throughput_kbps': throughput,
                'latency_s': latency
            })

        return pd.DataFrame(df_metrics)

    def plot_metrics(self, df_metrics, save_name='metrics_total.png'):
        if df_metrics.empty:
            print("Empty metrics — nothing to plot")
            return

        fig, axes = plt.subplots(2,2,figsize=(18,12))

        def plot_line(ax, x, y, title, ylabel, color, percent=False):
            ax.plot(x,y,marker='o',color=color,linewidth=2)
            ax.set_title(title,fontweight='bold')
            ax.set_xlabel("Number of IoT Nodes",fontweight='bold')
            ax.set_ylabel(ylabel,fontweight='bold')
            ax.grid(True,alpha=0.3)
            for xi, yi in zip(x,y):
                val = yi*100 if percent else yi
                fmt = "{:.1f}%" if percent else "{:.3f}" if ylabel=="Latency (s)" else "{:.0f}"
                ax.annotate(fmt.format(val),(xi,yi),textcoords="offset points",xytext=(0,10),
                            ha='center',fontsize=10,fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3",facecolor='white',alpha=0.8))

        plot_line(axes[0,0], df_metrics['iot_nodes'], df_metrics['latency_s'], "Latency vs IoT Nodes","Latency (s)","#E74C3C")
        plot_line(axes[0,1], df_metrics['iot_nodes'], df_metrics['throughput_kbps'], "Throughput vs IoT Nodes","Throughput (Kbps)","#2ECC71")
        plot_line(axes[1,0], df_metrics['iot_nodes'], df_metrics['packet_delivery_ratio'], "Packet Delivery Ratio vs IoT Nodes","Packet Delivery Ratio (%)","#3498DB",percent=True)
        plot_line(axes[1,1], df_metrics['iot_nodes'], df_metrics['accuracy'], "Accuracy vs IoT Nodes","Accuracy (%)","#9B59B6",percent=True)

        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, save_name)
        plt.savefig(save_path,dpi=300)
        plt.close()
        print(f"Plot saved: {save_path}")
        return save_path


if __name__ == "__main__":
    analyzer = AccurateNS3AnalyzerV9(model_path=MODEL_PATH)
    df_metrics = analyzer.analyze_all_files()
    print("\nMetrics summary:\n", df_metrics)
    analyzer.plot_metrics(df_metrics)
