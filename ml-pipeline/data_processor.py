import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NS3DataProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.base_path = Path(config['project']['base_path'])
        
    def process_ns3_data(self, csv_file):
        """Process NS-3 simulation data and extract features"""
        self.logger.info(f"Processing NS-3 data from {csv_file}")
        
        try:
            # Read NS-3 results
            df = pd.read_csv(csv_file)
            
            # Extract features from flow statistics
            features = []
            labels = []
            
            for _, flow in df.iterrows():
                flow_features = self._extract_flow_features(flow)
                if flow_features:
                    features.append(flow_features)
                    labels.append(flow['label'])
            
            return np.array(features), np.array(labels)
            
        except Exception as e:
            self.logger.error(f"Error processing NS-3 data: {e}")
            return None, None
    
    def _extract_flow_features(self, flow):
        """Extract comprehensive features from flow data"""
        try:
            features = [
                flow['tx_packets'],           # Total packets sent
                flow['rx_packets'],           # Total packets received  
                flow['tx_bytes'],             # Total bytes sent
                flow['rx_bytes'],             # Total bytes received
                flow['delay_sum'],            # Total delay
                flow['loss_rate'],            # Packet loss rate
                
                # Derived features
                flow['tx_bytes'] / max(1, flow['tx_packets']),  # Avg packet size
                flow['tx_packets'] / 60.0,   # Packet rate (packets/sec)
                flow['tx_bytes'] / 60.0,     # Byte rate (bytes/sec)
                flow['rx_packets'] / max(1, flow['tx_packets']),  # Delivery ratio
                
                # Statistical features
                flow['tx_packets'] - flow['rx_packets'],  # Lost packets
                flow['delay_sum'] / max(1, flow['rx_packets']),  # Avg delay per packet
            ]
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Error extracting features from flow: {e}")
            return None
    
    def generate_realistic_data(self, node_count, attacker_count):
        """Generate realistic data based on NS-3 simulation patterns"""
        self.logger.info(f"Generating realistic data for {node_count} nodes, {attacker_count} attackers")
        
        np.random.seed(42)
        n_samples = node_count * 10  # Multiple flows per node
        
        features = []
        labels = []
        
        # Normal traffic flows
        for _ in range(n_samples - attacker_count * 20):
            normal_flow = self._generate_normal_flow()
            features.append(normal_flow)
            labels.append(0)
        
        # Attack traffic flows  
        for _ in range(attacker_count * 20):
            attack_flow = self._generate_attack_flow()
            features.append(attack_flow)
            labels.append(1)
        
        return np.array(features), np.array(labels)
    
    def _generate_normal_flow(self):
        """Generate features for normal network traffic"""
        return [
            np.random.normal(100, 30),    # tx_packets
            np.random.normal(95, 28),     # rx_packets  
            np.random.normal(10000, 3000), # tx_bytes
            np.random.normal(9500, 2800),  # rx_bytes
            np.random.normal(5, 2),       # delay_sum
            np.random.normal(0.05, 0.02), # loss_rate
            np.random.normal(100, 20),    # avg_packet_size
            np.random.normal(2, 0.5),     # packet_rate
            np.random.normal(200, 50),    # byte_rate
            np.random.normal(0.95, 0.03), # delivery_ratio
            np.random.normal(5, 3),       # lost_packets
            np.random.normal(0.05, 0.02)  # avg_delay_per_packet
        ]
    
    def _generate_attack_flow(self):
        """Generate features for DDoS attack traffic"""
        return [
            np.random.normal(1000, 300),   # tx_packets (high)
            np.random.normal(500, 200),    # rx_packets (low due to loss)
            np.random.normal(500000, 150000), # tx_bytes (high)
            np.random.normal(250000, 100000), # rx_bytes (low)
            np.random.normal(50, 20),      # delay_sum (high)
            np.random.normal(0.5, 0.2),    # loss_rate (high)
            np.random.normal(500, 100),    # avg_packet_size
            np.random.normal(20, 5),       # packet_rate (high)
            np.random.normal(10000, 3000), # byte_rate (high)
            np.random.normal(0.5, 0.2),    # delivery_ratio (low)
            np.random.normal(500, 200),    # lost_packets (high)
            np.random.normal(0.1, 0.05)    # avg_delay_per_packet (high)
        ]
    
    def prepare_training_data(self, features, labels):
        """Prepare data for training"""
        if features is None or len(features) == 0:
            self.logger.warning("No features available, generating sample data")
            features, labels = self.generate_realistic_data(20, 5)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        self.logger.info(f"Training data prepared: {X_train.shape} train, {X_test.shape} test")
        
        return X_train, X_test, y_train, y_test