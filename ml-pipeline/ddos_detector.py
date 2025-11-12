import numpy as np
import joblib
import time
import pandas as pd
from scapy.all import sniff, IP, TCP, UDP, ICMP
import threading
from collections import defaultdict, deque
import warnings
import json
from datetime import datetime
import subprocess
import os
import socket

warnings.filterwarnings('ignore')

class RealTimeDDoSDetector:
    def __init__(self, model_path, threshold=0.8, config=None):
        self.config = config or {}
        self.load_model(model_path)
        self.threshold = threshold
        self.packet_buffer = deque(maxlen=1000)
        self.stats_window = deque(maxlen=100)
        self.attack_detected = False
        self.mitigation_active = False
        self.stats_lock = threading.Lock()
        self.detection_history = deque(maxlen=500)
        
        # Thống kê theo IP
        self.ip_stats = defaultdict(lambda: {
            'packet_count': 0,
            'byte_count': 0,
            'last_seen': 0,
            'ports': set(),
            'protocols': set(),
            'first_seen': time.time(),
            'is_blocked': False
        })
        
        # Thống kê tổng quan
        self.global_stats = {
            'total_packets': 0,
            'total_attack_packets': 0,
            'attack_start_time': None,
            'last_mitigation_action': None,
            'blocked_ips': set()
        }
        
        # Cấu hình mitigation
        self.mitigation_config = {
            'block_threshold': 100,
            'port_scan_threshold': 20,
            'max_blocked_ips': 50,
            'rate_limit_normal': '1000kbps',
            'rate_limit_suspicious': '100kbps',
            'check_interval': 5
        }
        
        # Cập nhật config nếu có
        if config and 'detection' in config:
            self.mitigation_config.update(config['detection'].get('mitigation', {}))
        
        print("🚀 DDoS Detection System Initialized")
        print(f"   Model: {self.model_metadata.get('best_model', 'Unknown')}")
        print(f"   Threshold: {self.threshold}")
        print(f"   Mitigation: Active")
    
    def load_model(self, model_path):
        """Load model đã train"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data.get('scaler', None)
            self.feature_names = model_data.get('feature_names', [])
            self.model_metadata = model_data
            print(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def extract_features(self, packet):
        """Trích xuất features từ packet network"""
        features = np.zeros(15)  # 15 features như trong training
        
        try:
            if IP in packet:
                # Basic packet features
                features[0] = len(packet)                    # packet_length
                features[1] = packet[IP].proto               # protocol
                features[2] = packet[IP].ttl                 # TTL
                
                # Protocol specific features
                if TCP in packet:
                    features[3] = packet[TCP].flags          # TCP flags
                    features[4] = packet[TCP].sport          # source_port
                    features[5] = packet[TCP].dport          # destination_port
                    features[1] = 6  # TCP protocol number
                elif UDP in packet:
                    features[3] = 0                          # No flags for UDP
                    features[4] = packet[UDP].sport
                    features[5] = packet[UDP].dport
                    features[1] = 17  # UDP protocol number
                elif ICMP in packet:
                    features[3] = 0
                    features[4] = 0
                    features[5] = 0
                    features[1] = 1  # ICMP protocol number
                else:
                    features[3] = 0
                    features[4] = 0
                    features[5] = 0
                
                # Statistical features (sẽ được cập nhật sau)
                features[6] = 0  # packet_rate
                features[7] = len(packet)  # byte_rate
                features[8] = 0  # packet_size_variance
                features[9] = 0  # port_entropy
                features[10] = 0  # flow_duration
                features[11] = 0  # time_between_packets
                features[12] = 0  # burst_detection
                features[13] = 0  # seasonal_patterns
                features[14] = 0  # connection_rate
                
        except Exception as e:
            print(f"Warning: Error extracting features from packet: {e}")
        
        return features
    
    def update_stats(self, packet):
        """Cập nhật thống kê real-time"""
        if IP in packet:
            src_ip = packet[IP].src
            current_time = time.time()
            
            with self.stats_lock:
                # Update IP statistics
                self.ip_stats[src_ip]['packet_count'] += 1
                self.ip_stats[src_ip]['byte_count'] += len(packet)
                self.ip_stats[src_ip]['last_seen'] = current_time
                
                if TCP in packet:
                    self.ip_stats[src_ip]['ports'].add(packet[TCP].sport)
                    self.ip_stats[src_ip]['protocols'].add('TCP')
                elif UDP in packet:
                    self.ip_stats[src_ip]['ports'].add(packet[UDP].sport)
                    self.ip_stats[src_ip]['protocols'].add('UDP')
                elif ICMP in packet:
                    self.ip_stats[src_ip]['protocols'].add('ICMP')
                
                # Update global statistics
                self.global_stats['total_packets'] += 1
    
    def calculate_real_time_features(self, src_ip):
        """Tính toán features real-time dựa trên thống kê"""
        with self.stats_lock:
            stats = self.ip_stats[src_ip]
            current_time = time.time()
            
            # Tính toán các features động
            features = np.zeros(15)
            
            # Basic features từ packet
            features[0] = stats['byte_count'] / max(1, stats['packet_count'])  # avg_packet_length
            features[1] = 6 if 'TCP' in stats['protocols'] else 17  # protocol
            features[2] = 64  # default TTL
            
            # Statistical features
            time_window = current_time - stats['first_seen']
            features[6] = stats['packet_count'] / max(1, time_window)  # packet_rate
            features[7] = stats['byte_count'] / max(1, time_window)    # byte_rate
            features[9] = len(stats['ports'])  # port_entropy (simplified)
            features[10] = time_window  # flow_duration
            features[14] = len(self.ip_stats)  # connection_rate
            
            return features
    
    def analyze_traffic_patterns(self):
        """Phân tích pattern traffic để phát hiện DDoS"""
        with self.stats_lock:
            if len(self.ip_stats) == 0:
                return False, []
            
            suspicious_ips = []
            total_packets = self.global_stats['total_packets']
            
            for ip, stats in self.ip_stats.items():
                if stats['is_blocked']:
                    continue
                
                suspicious_score = 0
                
                # High packet rate
                time_active = time.time() - stats['first_seen']
                packet_rate = stats['packet_count'] / max(1, time_active)
                if packet_rate > 100:  # packets per second
                    suspicious_score += 2
                
                # Port scanning behavior
                if len(stats['ports']) > self.mitigation_config['port_scan_threshold']:
                    suspicious_score += 3
                
                # Multiple protocols
                if len(stats['protocols']) > 2:
                    suspicious_score += 1
                
                # Short duration high activity
                if time_active < 10 and stats['packet_count'] > 500:
                    suspicious_score += 2
                
                if suspicious_score >= 3:
                    suspicious_ips.append((ip, suspicious_score))
            
            # Sắp xếp theo mức độ nghi ngờ
            suspicious_ips.sort(key=lambda x: x[1], reverse=True)
            return len(suspicious_ips) > 0, suspicious_ips
    
    def packet_handler(self, packet):
        """Xử lý mỗi packet nhận được"""
        try:
            # Cập nhật thống kê
            self.update_stats(packet)
            
            # Trích xuất features cơ bản
            features = self.extract_features(packet)
            
            # Dự đoán với ML model
            if hasattr(packet, 'src'):
                src_ip = packet[IP].src
                real_time_features = self.calculate_real_time_features(src_ip)
                
                # Kết hợp features
                combined_features = np.concatenate([features, real_time_features])
                final_features = combined_features[:15]  # Lấy 15 features đầu
                
                # Chuẩn hóa features nếu có scaler
                if self.scaler:
                    final_features = self.scaler.transform([final_features])
                
                prediction = self.model.predict([final_features])[0]
                probability = self.model.predict_proba([final_features])[0][1]
                
                # Lưu kết quả detection
                detection_record = {
                    'timestamp': time.time(),
                    'src_ip': src_ip,
                    'prediction': prediction,
                    'probability': probability,
                    'packet_length': len(packet)
                }
                self.detection_history.append(detection_record)
                
                # Kiểm tra DDoS
                ml_detection = probability > self.threshold
                pattern_detection, suspicious_ips = self.analyze_traffic_patterns()
                
                current_attack_state = ml_detection or (pattern_detection and len(suspicious_ips) > 2)
                
                if current_attack_state and not self.attack_detected:
                    self.attack_detected = True
                    self.global_stats['attack_start_time'] = time.time()
                    print(f"\n🚨 CRITICAL: DDoS ATTACK DETECTED!")
                    print(f"   Timestamp: {datetime.now().isoformat()}")
                    print(f"   ML Confidence: {probability:.4f}")
                    print(f"   Suspicious IPs: {len(suspicious_ips)}")
                    print(f"   Total Packets: {self.global_stats['total_packets']}")
                    self.trigger_mitigation(suspicious_ips)
                
                elif not current_attack_state and self.attack_detected:
                    self.attack_detected = False
                    print(f"\n✅ ATTACK MITIGATED - System returning to normal")
                    self.mitigation_active = False
                
                # Hiển thị thông tin real-time
                if len(self.detection_history) % 100 == 0:
                    self.print_realtime_stats()
                    
        except Exception as e:
            print(f"Error processing packet: {e}")
    
    def trigger_mitigation(self, suspicious_ips):
        """Kích hoạt biện pháp giảm thiểu"""
        print("🛡️  ACTIVATING MITIGATION MEASURES...")
        self.mitigation_active = True
        
        # 1. Block suspicious IPs
        blocked_count = 0
        for ip, score in suspicious_ips[:self.mitigation_config['max_blocked_ips']]:
            if self.block_ip(ip):
                blocked_count += 1
                self.global_stats['blocked_ips'].add(ip)
                with self.stats_lock:
                    self.ip_stats[ip]['is_blocked'] = True
        
        # 2. Rate limiting
        print("   Implementing rate limiting...")
        self.apply_rate_limiting()
        
        # 3. Alert system
        print("   Sending alerts to administrator...")
        self.send_alerts(blocked_count, len(suspicious_ips))
        
        # 4. Log mitigation action
        self.global_stats['last_mitigation_action'] = time.time()
        
        print(f"   ✅ Mitigation completed: {blocked_count} IPs blocked")
    
    def block_ip(self, ip_address):
        """Chặn IP address using iptables"""
        try:
            if os.name == 'posix':  # Linux/Unix
                # Check if rule already exists
                check_cmd = f"sudo iptables -C INPUT -s {ip_address} -j DROP 2>/dev/null"
                result = subprocess.run(check_cmd, shell=True, capture_output=True)
                
                if result.returncode != 0:  # Rule doesn't exist
                    block_cmd = f"sudo iptables -A INPUT -s {ip_address} -j DROP"
                    subprocess.run(block_cmd, shell=True, check=True)
                    print(f"   🔒 Blocked IP: {ip_address}")
                    return True
                else:
                    print(f"   ⚠️  IP already blocked: {ip_address}")
                    return False
            else:
                print(f"   📝 Would block IP: {ip_address} (Windows/other OS)")
                return True
                
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error blocking IP {ip_address}: {e}")
            return False
        except Exception as e:
            print(f"   ❌ Error blocking IP {ip_address}: {e}")
            return False
    
    def apply_rate_limiting(self):
        """Áp dụng rate limiting cho network traffic"""
        try:
            if os.name == 'posix':
                # Basic rate limiting example
                rate_cmd = "sudo iptables -A INPUT -p tcp --dport 80 -m limit --limit 100/minute --limit-burst 200 -j ACCEPT"
                subprocess.run(rate_cmd, shell=True, check=True)
                print("   📊 Rate limiting rules applied")
        except Exception as e:
            print(f"   ❌ Error applying rate limiting: {e}")
    
    def send_alerts(self, blocked_count, suspicious_count):
        """Gửi cảnh báo đến administrator"""
        alert_message = f"""
🚨 DDoS ATTACK ALERT 🚨

Time: {datetime.now().isoformat()}
Status: Attack in progress
Blocked IPs: {blocked_count}
Suspicious IPs: {suspicious_count}
Total Packets: {self.global_stats['total_packets']}
Attack Start: {self.global_stats['attack_start_time']}

Mitigation actions have been activated.
"""
        print(alert_message)
        
        # Có thể tích hợp với email, SMS, hoặc hệ thống cảnh báo khác
        # self.send_email_alert(alert_message)
        # self.send_slack_alert(alert_message)
    
    def print_realtime_stats(self):
        """Hiển thị thống kê real-time"""
        with self.stats_lock:
            total_ips = len(self.ip_stats)
            blocked_ips = len([ip for ip, stats in self.ip_stats.items() if stats['is_blocked']])
            attack_packets = sum(1 for record in self.detection_history if record['prediction'] == 1)
            
            print(f"\n📊 Real-time Stats:")
            print(f"   Total Packets: {self.global_stats['total_packets']}")
            print(f"   Attack Packets: {attack_packets}")
            print(f"   Unique IPs: {total_ips}")
            print(f"   Blocked IPs: {blocked_ips}")
            print(f"   Attack Status: {'ACTIVE' if self.attack_detected else 'Normal'}")
            print(f"   Mitigation: {'ACTIVE' if self.mitigation_active else 'Inactive'}")
    
    def save_detection_log(self):
        """Lưu log detection vào file"""
        log_data = {
            'detection_history': list(self.detection_history),
            'global_stats': self.global_stats,
            'ip_stats': dict(self.ip_stats),
            'end_time': time.time()
        }
        
        log_file = f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"📝 Detection log saved to {log_file}")
    
    def start_monitoring(self, interface=None, count=0):
        """Bắt đầu giám sát network traffic"""
        print("🚀 Starting DDoS Detection System...")
        print("📡 Monitoring network traffic...")
        print("   Press Ctrl+C to stop monitoring\n")
        
        try:
            # Bắt đầu packet sniffing
            sniff(
                prn=self.packet_handler,
                iface=interface,
                store=False,
                count=count  # 0 = unlimited
            )
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping DDoS detection system...")
            self.save_detection_log()
            print("✅ System stopped gracefully")
            
        except Exception as e:
            print(f"❌ Error starting monitor: {e}")
            self.save_detection_log()

# Utility function để test detection system
def test_detection_system():
    """Test the detection system với dữ liệu mẫu"""
    detector = RealTimeDDoSDetector('models/ddos_model.pkl')
    
    print("🧪 Testing detection system...")
    
    # Tạo packet mẫu để test
    from scapy.all import Ether, IP, TCP
    
    # Normal packet
    normal_packet = Ether()/IP(src="192.168.1.100", dst="10.0.0.1")/TCP(sport=12345, dport=80)
    detector.packet_handler(normal_packet)
    
    # Suspicious packet (high rate)
    for i in range(10):
        suspicious_packet = Ether()/IP(src="192.168.1.200", dst="10.0.0.1")/TCP(sport=10000+i, dport=80)
        detector.packet_handler(suspicious_packet)
    
    detector.print_realtime_stats()

if __name__ == "__main__":
    # Load config
    import yaml
    try:
        with open('../config/ml-config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except:
        config = {}
    
    # Khởi tạo detector
    detector = RealTimeDDoSDetector('../models/ddos_model.pkl', config=config)
    
    # Test mode hoặc real monitoring
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_detection_system()
    else:
        # Bắt đầu monitoring
        interface = sys.argv[1] if len(sys.argv) > 1 else None
        detector.start_monitoring(interface=interface)