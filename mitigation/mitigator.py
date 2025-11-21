import pandas as pd
import joblib
import time
import os
import sys
import warnings
from sklearn.preprocessing import StandardScaler

# Tắt cảnh báo feature names không khớp (để log sạch hơn)
warnings.filterwarnings("ignore", category=UserWarning)

# --- CẤU HÌNH ---
BASE_DIR = '/home/traphan/ns-3-dev/ddos-project-new'
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'ddos_model.pkl')

LIVE_DATA_DIR = os.path.join(BASE_DIR, 'data', 'live')
LIVE_STATS_FILE = os.path.join(LIVE_DATA_DIR, 'live_flow_stats.csv')
BLACKLIST_FILE = os.path.join(LIVE_DATA_DIR, 'blacklist.txt')

# Danh sách ĐẦY ĐỦ 11 đặc trưng (Cần để map dữ liệu ban đầu)
FULL_FEATURE_COLUMNS = [
    'protocol', 'tx_packets', 'rx_packets', 'tx_bytes', 'rx_bytes',
    'delay_sum', 'jitter_sum', 'lost_packets', 'packet_loss_ratio',
    'throughput', 'flow_duration'
]
# --------------------

class RealTimeMitigator:
    def __init__(self, model_path, stats_file, blacklist_file):
        print("--- Khởi tạo hệ thống giảm thiểu (Mitigation System) ---")
        self.stats_file = stats_file
        self.blacklist_file = blacklist_file
        
        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)

        # Load Model, Scaler và DANH SÁCH FEATURE QUAN TRỌNG
        self.model, self.scaler, self.selected_features = self._load_model(model_path)
        
        self.last_known_flows = set()
        self.blocked_ips = set()

        if os.path.exists(self.blacklist_file):
            try:
                os.remove(self.blacklist_file)
                print(f"Đã xóa blacklist cũ: {self.blacklist_file}")
            except OSError:
                pass

        print(f"✅ Model đã tải. Mô hình sử dụng {len(self.selected_features)} đặc trưng: {self.selected_features}")
        print("✅ Hệ thống đã sẵn sàng. Chờ dữ liệu từ NS-3...")

    def _load_model(self, path):
        try:
            if not os.path.exists(path):
                print(f"❌ Lỗi: Không tìm thấy file model tại {path}")
                sys.exit(1)
                
            data = joblib.load(path)
            model = data['model']
            scaler = data['scaler']
            
            # Lấy danh sách feature mà model đã học (Được lưu lúc train)
            # Nếu không có key này (model cũ), mặc định dùng full
            selected_features = data.get('feature_names', FULL_FEATURE_COLUMNS)
            
            return model, scaler, selected_features
        except Exception as e:
            print(f"❌ Lỗi tải model: {e}")
            sys.exit(1)

    def _normalize_dataframe(self, df):
        """Chuẩn hóa tên cột và dữ liệu"""
        df.columns = df.columns.str.strip()

        rename_map = {
            'src_ip': 'source_ip',
            'sourceAddress': 'source_ip',
        }
        df = df.rename(columns=rename_map)

        # Điền 0 vào các cột thiếu
        for col in FULL_FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0

        df = df.fillna(0)
        return df

    def _process_new_flows(self, new_flows_df):
        if new_flows_df.empty:
            return

        try:
            # BƯỚC 1: Lấy đủ 11 cột để đưa vào Scaler (Vì Scaler được fit trên 11 cột)
            X_full = new_flows_df[FULL_FEATURE_COLUMNS]
            
            # BƯỚC 2: Chuẩn hóa dữ liệu (Scaling)
            # Kết quả trả về là numpy array (mất tên cột)
            if self.scaler:
                X_scaled_array = self.scaler.transform(X_full)
            else:
                X_scaled_array = X_full.values

            # BƯỚC 3: Chuyển lại thành DataFrame để có tên cột
            X_scaled_df = pd.DataFrame(X_scaled_array, columns=FULL_FEATURE_COLUMNS)
            
            # BƯỚC 4: LỌC CỘT - Chỉ lấy đúng những cột mà Model cần (3 cột)
            # Đây là bước quan trọng để sửa lỗi mismatch
            X_final = X_scaled_df[self.selected_features]

            # BƯỚC 5: Dự đoán
            predictions = self.model.predict(X_final)
            new_flows_df['prediction'] = predictions

            # Lọc ra các flow tấn công (Label = 1)
            attack_flows = new_flows_df[new_flows_df['prediction'] == 1]

            if attack_flows.empty:
                return

            # Ghi vào Blacklist
            with open(self.blacklist_file, 'a') as f:
                for ip in attack_flows['source_ip'].unique():
                    if ip not in self.blocked_ips:
                        print(f"🚨 PHÁT HIỆN TẤN CÔNG từ IP: {ip} -> Đang chặn...")
                        f.write(f"{ip}\n")
                        f.flush()
                        self.blocked_ips.add(ip)
                        
        except Exception as e:
            print(f"⚠️ Lỗi khi dự đoán: {e}")
            # In chi tiết lỗi để debug nếu cần
            # import traceback
            # traceback.print_exc()

    def watch(self):
        print(f"DEBUG: Đang theo dõi file: {self.stats_file}")

        while not os.path.exists(self.stats_file):
            print(f"DEBUG: Đang chờ file {os.path.basename(self.stats_file)} được tạo...")
            time.sleep(1)

        print("DEBUG: File đã xuất hiện. Bắt đầu phân tích...")
        
        last_pos = 0

        while True:
            try:
                # Đọc file thông minh (chỉ đọc phần mới)
                with open(self.stats_file, 'r') as f:
                    f.seek(last_pos)
                    lines = f.readlines()
                    new_pos = f.tell()
                
                if new_pos == last_pos:
                    time.sleep(0.5)
                    continue
                
                # File bị reset (khi chạy lại mô phỏng mới)
                if new_pos < last_pos:
                    last_pos = 0
                    continue
                
                last_pos = new_pos
                
                if lines:
                    # Lọc bỏ header nếu nó xuất hiện lại giữa file
                    valid_lines = [line for line in lines if "time,source_ip" not in line]
                    if not valid_lines:
                        continue

                    from io import StringIO
                    csv_data = "".join(valid_lines)
                    
                    # Header chuẩn khớp với file C++
                    header_names = ["time","source_ip","protocol","tx_packets","rx_packets","tx_bytes","rx_bytes","delay_sum","jitter_sum","lost_packets","packet_loss_ratio","throughput","flow_duration","label"]
                    
                    df = pd.read_csv(StringIO(csv_data), names=header_names, on_bad_lines='skip')
                    
                    df = self._normalize_dataframe(df)
                    
                    # Tạo ID duy nhất: Thời gian + IP
                    df['record_id'] = df['time'].astype(str) + "-" + df['source_ip'].astype(str)
                    
                    # Lọc bản ghi mới
                    new_flows_df = df[~df['record_id'].isin(self.last_known_flows)].copy()

                    if not new_flows_df.empty:
                        # print(f"DEBUG: Nhận {len(new_flows_df)} dòng dữ liệu mới.")
                        self._process_new_flows(new_flows_df)
                        
                        # Update cache
                        self.last_known_flows.update(new_flows_df['record_id'])
                        
                        # Dọn dẹp cache nếu quá lớn
                        if len(self.last_known_flows) > 50000:
                            self.last_known_flows.clear()

            except Exception as e:
                print(f"❌ Lỗi vòng lặp chính: {e}")
                time.sleep(1)

if __name__ == "__main__":
    mitigator = RealTimeMitigator(
        model_path=MODEL_PATH,
        stats_file=LIVE_STATS_FILE,
        blacklist_file=BLACKLIST_FILE
    )
    mitigator.watch()