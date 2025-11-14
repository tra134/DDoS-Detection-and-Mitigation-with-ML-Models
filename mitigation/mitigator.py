import pandas as pd
import joblib
import time
import os
from sklearn.preprocessing import StandardScaler

# --- CẤU HÌNH ---
# Đường dẫn tuyệt đối
BASE_DIR = '/home/traphan/ns-3-dev/ddos-project-new'
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'ddos_model.pkl')

# <<< ĐƯỜNG DẪN ĐÚNG >>>
LIVE_DATA_DIR = os.path.join(BASE_DIR, 'data', 'live')
LIVE_STATS_FILE = os.path.join(LIVE_DATA_DIR, 'live_flow_stats.csv')
BLACKLIST_FILE = os.path.join(LIVE_DATA_DIR, 'blacklist.txt')

# Các đặc trưng (phải khớp 100% với lúc huấn luyện)
FEATURE_COLUMNS = [
    'protocol', 'tx_packets', 'rx_packets', 'tx_bytes', 'rx_bytes',
    'delay_sum', 'jitter_sum', 'lost_packets', 'packet_loss_ratio',
    'throughput', 'flow_duration'
]
# --- KẾT THÚC CẤU HÌNH ---

class RealTimeMitigator:
    def __init__(self, model_path, stats_file, blacklist_file):
        print("--- Khởi tạo Hệ thống Giảm thiểu (Mitigation System) ---")
        self.stats_file = stats_file
        self.blacklist_file = blacklist_file
        
        # Tự động tạo thư mục data/live
        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
        
        self.model, self.scaler = self._load_model(model_path)
        
        self.last_known_flows = set() 
        self.blocked_ips = set() 
        
        if os.path.exists(self.blacklist_file):
            os.remove(self.blacklist_file)
            print(f"Đã xóa file blacklist cũ: {self.blacklist_file}")
            
        print("✅ Bộ não AI đã sẵn sàng. Đang chờ dữ liệu từ NS-3...")

    def _load_model(self, path):
        """Tải mô hình và scaler đã lưu."""
        try:
            data = joblib.load(path)
            model = data['model']
            scaler = data['scaler']
            print(f"Tải thành công mô hình: {type(model).__name__}")
            print(f"Tải thành công scaler.")
            return model, scaler
        except FileNotFoundError:
            print(f"❌ LỖI: Không tìm thấy file model tại: {path}")
            print("Vui lòng chạy 'train_model.py' trước.")
            exit(1)
        except Exception as e:
            print(f"❌ LỖI: Không thể tải model: {e}")
            exit(1)

    def _process_new_flows(self, new_flows_df):
        """Phân tích các flow mới và ra quyết định chặn."""
        
        if new_flows_df.empty:
            return
            
        X = new_flows_df[FEATURE_COLUMNS]
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        
        new_flows_df['prediction'] = predictions
        
        attack_flows = new_flows_df[new_flows_df['prediction'] == 1]
        
        if attack_flows.empty:
            return

        with open(self.blacklist_file, 'a') as f:
            for ip in attack_flows['source_ip']:
                if ip not in self.blocked_ips:
                    print(f"🚨 PHÁT HIỆN TẤN CÔNG! IP: {ip}. Ghi vào blacklist...")
                    f.write(f"{ip}\n")
                    self.blocked_ips.add(ip)

    def watch(self):
        """Vòng lặp chính: Liên tục theo dõi file stats."""
        
        # <<< DÒNG DEBUG SỐ 1 (ĐÃ THÊM) >>>
        print(f"\nDEBUG: Đang theo dõi file tại: {self.stats_file}\n") 
            
        while True:
            try:
                # Vòng lặp chờ file
                while not os.path.exists(self.stats_file):
                    
                    # <<< DÒNG DEBUG SỐ 2 (ĐÃ THÊM) >>>
                    print(f"DEBUG: Đang chờ... (file {os.path.basename(self.stats_file)} chưa tồn tại)") 
                    
                    time.sleep(1)
                
                # File đã tồn tại, bắt đầu đọc
                df = pd.read_csv(self.stats_file, on_bad_lines='skip')
                
                if df.empty:
                    time.sleep(0.5)
                    continue

                df['flow_id'] = df['time'].astype(str) + '-' + df['source_ip']
                
                # Sửa lỗi SettingWithCopyWarning
                new_flows_df = df[~df['flow_id'].isin(self.last_known_flows)].copy()

                if not new_flows_df.empty:
                    # <<< DÒNG DEBUG SỐ 3 (ĐÃ THÊM) >>>
                    print(f"DEBUG: Phát hiện {len(new_flows_df)} flow mới. Đang phân tích...")
                    self._process_new_flows(new_flows_df)
                    self.last_known_flows.update(new_flows_df['flow_id'])
                
                # Nếu không có flow mới, thì không in gì cả (để yên lặng)
                time.sleep(0.5)

            except pd.errors.EmptyDataError:
                # Lỗi này xảy ra khi C++ đang ghi dở
                time.sleep(0.5) 
            except Exception as e:
                print(f"Lỗi trong vòng lặp watch: {e}")
                time.sleep(2)


if __name__ == "__main__":
    mitigator = RealTimeMitigator(
        model_path=MODEL_PATH,
        stats_file=LIVE_STATS_FILE,
        blacklist_file=BLACKLIST_FILE
    )
    mitigator.watch()