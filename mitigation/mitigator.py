import pandas as pd
import joblib
import time
import os
from sklearn.preprocessing import StandardScaler

# --- CẤU HÌNH ---
# Đường dẫn tuyệt đối (SỬA NẾU CẦN)
BASE_DIR = '/home/traphan/ns-3-dev/ddos-project-new'
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'ddos_model.pkl')

# Các file giao tiếp (phải khớp với code C++)
LIVE_STATS_FILE = os.path.join(BASE_DIR, 'live_flow_stats.csv')
BLACKLIST_FILE = os.path.join(BASE_DIR, 'blacklist.txt')

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
        self.model, self.scaler = self._load_model(model_path)
        
        # Dùng set để lưu trữ các IP đã đọc, tránh đọc lại
        self.last_known_flows = set() 
        # Dùng set để lưu các IP đã chặn, tránh ghi file trùng lặp
        self.blocked_ips = set() 
        
        # Xóa file blacklist cũ (nếu có) khi khởi động
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
            
        # 1. Chuẩn bị dữ liệu
        X = new_flows_df[FEATURE_COLUMNS]
        X_scaled = self.scaler.transform(X)
        
        # 2. Dự đoán
        predictions = self.model.predict(X_scaled)
        
        # 3. Thêm cột dự đoán vào DF để lọc
        new_flows_df['prediction'] = predictions
        
        # 4. Lọc ra các flow bị dự đoán là tấn công (label=1)
        attack_flows = new_flows_df[new_flows_df['prediction'] == 1]
        
        if attack_flows.empty:
            return # Không có tấn công mới

        # 5. Ghi IP tấn công vào blacklist
        # Mở file ở chế độ 'a' (append - ghi nối tiếp)
        with open(self.blacklist_file, 'a') as f:
            for ip in attack_flows['source_ip']:
                # Chỉ ghi nếu IP này CHƯA từng bị chặn
                if ip not in self.blocked_ips:
                    print(f"🚨 PHÁT HIỆN TẤN CÔNG! IP: {ip}. Ghi vào blacklist...")
                    f.write(f"{ip}\n")
                    self.blocked_ips.add(ip) # Thêm vào set để không ghi lại

    def watch(self):
        """Vòng lặp chính: Liên tục theo dõi file stats."""
        while True:
            try:
                # Chờ file được tạo ra bởi NS-3
                while not os.path.exists(self.stats_file):
                    time.sleep(1)
                
                # Đọc file CSV
                # Thêm 'on_bad_lines' để bỏ qua các dòng đang được C++ ghi dở
                df = pd.read_csv(self.stats_file, on_bad_lines='skip')
                
                if df.empty:
                    time.sleep(0.5)
                    continue

                # Tạo một ID duy nhất cho mỗi flow (time + source_ip)
                # để biết flow nào là mới
                df['flow_id'] = df['time'].astype(str) + '-' + df['source_ip']
                
                # Lọc ra các flow_id CHƯA từng thấy
                new_flows_df = df[~df['flow_id'].isin(self.last_known_flows)]

                if not new_flows_df.empty:
                    # Xử lý các flow mới
                    self._process_new_flows(new_flows_df)
                    
                    # Cập nhật set các flow đã biết
                    self.last_known_flows.update(new_flows_df['flow_id'])
                
                # Nghỉ 0.5 giây trước khi kiểm tra lại
                time.sleep(0.5)

            except pd.errors.EmptyDataError:
                # Lỗi này xảy ra khi Python đọc file đúng lúc C++ đang xóa/ghi
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