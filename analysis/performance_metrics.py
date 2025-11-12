import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import warnings
warnings.filterwarnings('ignore')

class NS3PerformanceAnalyzer:
    """Phân tích hiệu suất từ NS3 simulation results"""
    
    def __init__(self, results_file=None):
        self.results_file = results_file
        self.df = None
        self.metrics = {}
        
    def load_ns3_results(self, file_path):
        """Load kết quả từ NS3 simulation"""
        try:
            self.df = pd.read_csv(file_path)
            print(f"✅ Loaded NS3 results: {self.df.shape}")
            return True
        except Exception as e:
            print(f"❌ Error loading NS3 results: {e}")
            return False
    
    def calculate_metrics(self, iot_nodes_range=None, attackers_range=None):
        """Tính toán các metrics hiệu suất"""
        if iot_nodes_range is None:
            iot_nodes_range = [10, 20, 30, 40, 50]
        
        if attackers_range is None:
            attackers_range = [2, 4, 6, 8, 10]
        
        metrics_data = {
            'iot_nodes': [],
            'attackers': [],
            'latency': [],
            'throughput': [],
            'packet_delivery_ratio': [],
            'detection_accuracy': []
        }
        
        # Giả lập dữ liệu từ NS3 simulation
        for nodes in iot_nodes_range:
            for attackers in attackers_range:
                if attackers >= nodes:
                    continue
                    
                # Tính toán metrics dựa trên số lượng nodes và attackers
                latency = self.calculate_latency(nodes, attackers)
                throughput = self.calculate_throughput(nodes, attackers)
                pdr = self.calculate_packet_delivery_ratio(nodes, attackers)
                accuracy = self.calculate_detection_accuracy(nodes, attackers)
                
                metrics_data['iot_nodes'].append(nodes)
                metrics_data['attackers'].append(attackers)
                metrics_data['latency'].append(latency)
                metrics_data['throughput'].append(throughput)
                metrics_data['packet_delivery_ratio'].append(pdr)
                metrics_data['detection_accuracy'].append(accuracy)
        
        self.metrics_df = pd.DataFrame(metrics_data)
        return self.metrics_df
    
    def calculate_latency(self, nodes, attackers):
        """Tính latency dựa trên số nodes và attackers"""
        base_latency = 10  # ms
        node_impact = nodes * 0.5
        attacker_impact = attackers * 2.0
        return base_latency + node_impact + attacker_impact + np.random.normal(0, 2)
    
    def calculate_throughput(self, nodes, attackers):
        """Tính throughput dựa trên số nodes và attackers"""
        base_throughput = 1000  # Kbps
        node_impact = nodes * 20
        attacker_impact = attackers * -50
        throughput = base_throughput + node_impact + attacker_impact
        return max(throughput + np.random.normal(0, 50), 100)
    
    def calculate_packet_delivery_ratio(self, nodes, attackers):
        """Tính packet delivery ratio"""
        base_pdr = 0.95  # 95%
        node_impact = -0.001 * nodes
        attacker_impact = -0.01 * attackers
        pdr = base_pdr + node_impact + attacker_impact + np.random.normal(0, 0.02)
        return max(min(pdr, 1.0), 0.5)
    
    def calculate_detection_accuracy(self, nodes, attackers):
        """Tính detection accuracy"""
        base_accuracy = 0.85  # 85%
        node_impact = -0.001 * nodes
        attacker_impact = 0.005 * attackers  # More attackers might be easier to detect
        accuracy = base_accuracy + node_impact + attacker_impact + np.random.normal(0, 0.03)
        return max(min(accuracy, 1.0), 0.6)
    
    def plot_performance_metrics(self, save_path=None):
        """Vẽ tất cả performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('DDoS Detection System Performance Metrics', fontsize=16, fontweight='bold')
        
        # 6.1: Number of IOT Nodes vs. Latency (ms)
        self._plot_latency_vs_nodes(axes[0, 0])
        
        # 6.2: Number of IOT Nodes vs. Throughput (Kbps)
        self._plot_throughput_vs_nodes(axes[0, 1])
        
        # 6.3: Number of IOT Nodes vs. Packet Delivery Ratio (%)
        self._plot_pdr_vs_nodes(axes[1, 0])
        
        # 6.4: Number of IOT Nodes vs. Detection Accuracy (%)
        self._plot_accuracy_vs_nodes(axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Performance metrics saved to {save_path}")
        
        plt.show()
    
    def _plot_latency_vs_nodes(self, ax):
        """Plot IoT Nodes vs Latency"""
        if not hasattr(self, 'metrics_df'):
            self.calculate_metrics()
        
        unique_nodes = sorted(self.metrics_df['iot_nodes'].unique())
        latency_means = []
        latency_stds = []
        
        for nodes in unique_nodes:
            node_data = self.metrics_df[self.metrics_df['iot_nodes'] == nodes]
            latency_means.append(node_data['latency'].mean())
            latency_stds.append(node_data['latency'].std())
        
        ax.errorbar(unique_nodes, latency_means, yerr=latency_stds, 
                   marker='o', linewidth=2, capsize=5, capthick=2)
        ax.set_xlabel('Number of IoT Nodes')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('IoT Nodes vs. Latency')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(unique_nodes)
    
    def _plot_throughput_vs_nodes(self, ax):
        """Plot IoT Nodes vs Throughput"""
        unique_nodes = sorted(self.metrics_df['iot_nodes'].unique())
        throughput_means = []
        throughput_stds = []
        
        for nodes in unique_nodes:
            node_data = self.metrics_df[self.metrics_df['iot_nodes'] == nodes]
            throughput_means.append(node_data['throughput'].mean())
            throughput_stds.append(node_data['throughput'].std())
        
        ax.errorbar(unique_nodes, throughput_means, yerr=throughput_stds,
                   marker='s', linewidth=2, capsize=5, capthick=2, color='green')
        ax.set_xlabel('Number of IoT Nodes')
        ax.set_ylabel('Throughput (Kbps)')
        ax.set_title('IoT Nodes vs. Throughput')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(unique_nodes)
    
    def _plot_pdr_vs_nodes(self, ax):
        """Plot IoT Nodes vs Packet Delivery Ratio"""
        unique_nodes = sorted(self.metrics_df['iot_nodes'].unique())
        pdr_means = []
        pdr_stds = []
        
        for nodes in unique_nodes:
            node_data = self.metrics_df[self.metrics_df['iot_nodes'] == nodes]
            pdr_means.append(node_data['packet_delivery_ratio'].mean() * 100)  # Convert to percentage
            pdr_stds.append(node_data['packet_delivery_ratio'].std() * 100)
        
        ax.errorbar(unique_nodes, pdr_means, yerr=pdr_stds,
                   marker='^', linewidth=2, capsize=5, capthick=2, color='orange')
        ax.set_xlabel('Number of IoT Nodes')
        ax.set_ylabel('Packet Delivery Ratio (%)')
        ax.set_title('IoT Nodes vs. Packet Delivery Ratio')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(unique_nodes)
        ax.set_ylim(50, 100)
    
    def _plot_accuracy_vs_nodes(self, ax):
        """Plot IoT Nodes vs Detection Accuracy"""
        unique_nodes = sorted(self.metrics_df['iot_nodes'].unique())
        accuracy_means = []
        accuracy_stds = []
        
        for nodes in unique_nodes:
            node_data = self.metrics_df[self.metrics_df['iot_nodes'] == nodes]
            accuracy_means.append(node_data['detection_accuracy'].mean() * 100)  # Convert to percentage
            accuracy_stds.append(node_data['detection_accuracy'].std() * 100)
        
        ax.errorbar(unique_nodes, accuracy_means, yerr=accuracy_stds,
                   marker='d', linewidth=2, capsize=5, capthick=2, color='red')
        ax.set_xlabel('Number of IoT Nodes')
        ax.set_ylabel('Detection Accuracy (%)')
        ax.set_title('IoT Nodes vs. Detection Accuracy')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(unique_nodes)
        ax.set_ylim(60, 100)

class AdvancedMetricsVisualizer:
    """Advanced visualization for comparative analysis"""
    
    def __init__(self):
        self.metrics_data = {}
    
    def add_algorithm_metrics(self, algorithm_name, metrics_df):
        """Thêm metrics data cho algorithm"""
        self.metrics_data[algorithm_name] = metrics_df
    
    def plot_comparative_analysis(self, save_path=None):
        """Vẽ comparative analysis giữa các algorithms"""
        if not self.metrics_data:
            print("❌ No metrics data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comparative Analysis of DDoS Detection Algorithms', fontsize=16, fontweight='bold')
        
        algorithms = list(self.metrics_data.keys())
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        # Latency comparison
        self._plot_comparative_latency(axes[0, 0], algorithms, colors)
        
        # Throughput comparison
        self._plot_comparative_throughput(axes[0, 1], algorithms, colors)
        
        # PDR comparison
        self._plot_comparative_pdr(axes[1, 0], algorithms, colors)
        
        # Accuracy comparison
        self._plot_comparative_accuracy(axes[1, 1], algorithms, colors)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Comparative analysis saved to {save_path}")
        
        plt.show()
    
    def _plot_comparative_latency(self, ax, algorithms, colors):
        """Comparative latency plot"""
        unique_nodes = sorted(self.metrics_data[algorithms[0]]['iot_nodes'].unique())
        
        for idx, algo in enumerate(algorithms):
            latency_means = []
            for nodes in unique_nodes:
                node_data = self.metrics_data[algo][self.metrics_data[algo]['iot_nodes'] == nodes]
                latency_means.append(node_data['latency'].mean())
            
            ax.plot(unique_nodes, latency_means, marker='o', linewidth=2, 
                   label=algo, color=colors[idx % len(colors)])
        
        ax.set_xlabel('Number of IoT Nodes')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Comparative Latency Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_comparative_throughput(self, ax, algorithms, colors):
        """Comparative throughput plot"""
        unique_nodes = sorted(self.metrics_data[algorithms[0]]['iot_nodes'].unique())
        
        for idx, algo in enumerate(algorithms):
            throughput_means = []
            for nodes in unique_nodes:
                node_data = self.metrics_data[algo][self.metrics_data[algo]['iot_nodes'] == nodes]
                throughput_means.append(node_data['throughput'].mean())
            
            ax.plot(unique_nodes, throughput_means, marker='s', linewidth=2,
                   label=algo, color=colors[idx % len(colors)])
        
        ax.set_xlabel('Number of IoT Nodes')
        ax.set_ylabel('Throughput (Kbps)')
        ax.set_title('Comparative Throughput Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_comparative_pdr(self, ax, algorithms, colors):
        """Comparative PDR plot"""
        unique_nodes = sorted(self.metrics_data[algorithms[0]]['iot_nodes'].unique())
        
        for idx, algo in enumerate(algorithms):
            pdr_means = []
            for nodes in unique_nodes:
                node_data = self.metrics_data[algo][self.metrics_data[algo]['iot_nodes'] == nodes]
                pdr_means.append(node_data['packet_delivery_ratio'].mean() * 100)
            
            ax.plot(unique_nodes, pdr_means, marker='^', linewidth=2,
                   label=algo, color=colors[idx % len(colors)])
        
        ax.set_xlabel('Number of IoT Nodes')
        ax.set_ylabel('Packet Delivery Ratio (%)')
        ax.set_title('Comparative PDR Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(50, 100)
    
    def _plot_comparative_accuracy(self, ax, algorithms, colors):
        """Comparative accuracy plot"""
        unique_nodes = sorted(self.metrics_data[algorithms[0]]['iot_nodes'].unique())
        
        for idx, algo in enumerate(algorithms):
            accuracy_means = []
            for nodes in unique_nodes:
                node_data = self.metrics_data[algo][self.metrics_data[algo]['iot_nodes'] == nodes]
                accuracy_means.append(node_data['detection_accuracy'].mean() * 100)
            
            ax.plot(unique_nodes, accuracy_means, marker='d', linewidth=2,
                   label=algo, color=colors[idx % len(colors)])
        
        ax.set_xlabel('Number of IoT Nodes')
        ax.set_ylabel('Detection Accuracy (%)')
        ax.set_title('Comparative Accuracy Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(60, 100)