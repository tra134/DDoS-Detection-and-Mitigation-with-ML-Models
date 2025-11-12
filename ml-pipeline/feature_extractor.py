import numpy as np
import pandas as pd
from collections import defaultdict, deque
import time

class FeatureExtractor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.packet_buffer = deque(maxlen=window_size)
        self.flow_stats = defaultdict(lambda: {
            'packet_count': 0,
            'byte_count': 0,
            'start_time': None,
            'last_seen': None,
            'ports': set(),
            'protocols': set()
        })
    
    def extract_flow_features(self, packet):
        """Trích xuất features theo flow"""
        features = {}
        
        if hasattr(packet, 'src') and hasattr(packet, 'dst'):
            flow_key = f"{packet.src}_{packet.dst}"
            
            # Update flow statistics
            self.update_flow_stats(flow_key, packet)
            flow_data = self.flow_stats[flow_key]
            
            # Tính toán flow features
            features = {
                'flow_duration': time.time() - flow_data['start_time'] if flow_data['start_time'] else 0,
                'packet_count': flow_data['packet_count'],
                'byte_count': flow_data['byte_count'],
                'avg_packet_size': flow_data['byte_count'] / max(1, flow_data['packet_count']),
                'port_entropy': len(flow_data['ports']),
                'protocol_count': len(flow_data['protocols']),
                'packet_rate': flow_data['packet_count'] / max(1, (time.time() - flow_data['start_time']))
            }
        
        return features
    
    def update_flow_stats(self, flow_key, packet):
        """Cập nhật thống kê flow"""
        current_time = time.time()
        flow_data = self.flow_stats[flow_key]
        
        if flow_data['start_time'] is None:
            flow_data['start_time'] = current_time
        
        flow_data['packet_count'] += 1
        flow_data['byte_count'] += len(packet)
        flow_data['last_seen'] = current_time
        
        if hasattr(packet, 'sport'):
            flow_data['ports'].add(packet.sport)
        if hasattr(packet, 'proto'):
            flow_data['protocols'].add(packet.proto)
    
    def cleanup_old_flows(self, timeout=300):
        """Dọn dẹp flows cũ"""
        current_time = time.time()
        expired_flows = []
        
        for flow_key, flow_data in self.flow_stats.items():
            if flow_data['last_seen'] and (current_time - flow_data['last_seen']) > timeout:
                expired_flows.append(flow_key)
        
        for flow_key in expired_flows:
            del self.flow_stats[flow_key]