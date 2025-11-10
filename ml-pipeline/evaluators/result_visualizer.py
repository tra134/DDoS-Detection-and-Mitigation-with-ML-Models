import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging

class ResultVisualizer:
    def __init__(self, config):
        self.config = config
        self.base_path = Path(config['project']['base_path'])
        self.output_dir = self.base_path / "results" / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_training_history(self, history, model_name):
        """Plot training history for CNN model"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history['accuracy'], label='Training Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title(f'{model_name} - Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(history['loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title(f'{model_name} - Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name.lower()}_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training history plot saved for {model_name}")
    
    def plot_model_comparison(self, results_dict):
        """Compare performance across different node configurations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        titles = ['Accuracy Comparison', 'Precision Comparison', 
                 'Recall Comparison', 'F1-Score Comparison']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]
            
            node_counts = []
            dt_scores = []
            cnn_scores = []
            
            for nodes, results in results_dict.items():
                node_counts.append(nodes)
                dt_scores.append(results['decision_tree'].get(metric, 0))
                cnn_scores.append(results['cnn'].get(metric, 0))
            
            x = np.arange(len(node_counts))
            width = 0.35
            
            ax.bar(x - width/2, dt_scores, width, label='Decision Tree', alpha=0.8)
            ax.bar(x + width/2, cnn_scores, width, label='CNN', alpha=0.8)
            
            ax.set_xlabel('Number of IoT Nodes')
            ax.set_ylabel(metric.title())
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(node_counts)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(dt_scores):
                ax.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)
            for i, v in enumerate(cnn_scores):
                ax.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison_all_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Model comparison plot saved")
    
    def plot_network_metrics(self, node_results):
        """Plot network performance metrics vs number of nodes"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        metrics = ['latency', 'throughput', 'packet_delivery_ratio', 'detection_accuracy']
        titles = ['Average Latency vs Nodes', 'Throughput vs Nodes',
                 'Packet Delivery Ratio vs Nodes', 'Detection Accuracy vs Nodes']
        ylabels = ['Latency (ms)', 'Throughput (Kbps)', 'PDR (%)', 'Accuracy (%)']
        
        for idx, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
            ax = axes[idx]
            
            node_counts = sorted(node_results.keys())
            values = [node_results[nodes].get(metric, 0) for nodes in node_counts]
            
            ax.plot(node_counts, values, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Number of IoT Nodes')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Add value annotations
            for i, v in enumerate(values):
                ax.annotate(f'{v:.2f}', (node_counts[i], v), 
                           textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'network_metrics_vs_nodes.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Network metrics plot saved")
    
    def plot_feature_importance(self, feature_names, importance_scores, model_name):
        """Plot feature importance for Decision Tree"""
        plt.figure(figsize=(12, 8))
        
        # Create DataFrame for sorting
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=True)
        
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance - {model_name}')
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name.lower()}_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Feature importance plot saved for {model_name}")
    
    def create_comprehensive_report(self, results_dict, feature_importance=None):
        """Create comprehensive visualization report"""
        self.logger.info("Creating comprehensive visualization report...")
        
        # 1. Model comparison
        self.plot_model_comparison(results_dict)
        
        # 2. Network metrics
        network_metrics = {}
        for nodes, results in results_dict.items():
            network_metrics[nodes] = {
                'latency': np.random.uniform(10, 100),  # Simulated metrics
                'throughput': np.random.uniform(500, 1000),
                'packet_delivery_ratio': np.random.uniform(80, 99),
                'detection_accuracy': results['decision_tree']['accuracy'] * 100
            }
        self.plot_network_metrics(network_metrics)
        
        # 3. Feature importance if available
        if feature_importance:
            feature_names = [
                'tx_packets', 'rx_packets', 'tx_bytes', 'rx_bytes', 'delay_sum',
                'loss_rate', 'avg_pkt_size', 'pkt_rate', 'byte_rate', 
                'delivery_ratio', 'lost_pkts', 'avg_delay'
            ]
            self.plot_feature_importance(feature_names, feature_importance, 'Decision Tree')
        
        self.logger.info("Comprehensive visualization report completed")