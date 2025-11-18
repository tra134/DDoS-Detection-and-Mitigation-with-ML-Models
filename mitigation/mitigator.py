import pandas as pd
import joblib
import time
import os
from sklearn.preprocessing import StandardScaler

# --- CẤU HÌNH ---
BASE_DIR = '/home/traphan/ns-3-dev/ddos-project-new'
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'ddos_model.pkl')

LIVE_DATA_DIR = os.path.join(BASE_DIR, 'data', 'live')
LIVE_STATS_FILE = os.path.join(LIVE_DATA_DIR, 'live_flow_stats.csv')
BLACKLIST_FILE = os.path.join(LIVE_DATA_DIR, 'blacklist.txt')

# Các đặc trưng phải trùng với model
FEATURE_COLUMNS = [
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

        self.model, self.scaler = self._load_model(model_path)
        self.last_known_flows = set()
        self.blocked_ips = set()

        if os.path.exists(self.blacklist_file):
            os.remove(self.blacklist_file)
            print(f"Đã xóa blacklist cũ: {self.blacklist_file}")

        print("✅ Hệ thống đã sẵn sàng. Chờ dữ liệu từ NS‑3...")

    def _load_model(self, path):
        try:
            data = joblib.load(path)
            return data['model'], data['scaler']
        except Exception as e:
            print(f"❌ Lỗi tải model: {e}")
            exit(1)

    # ===============================
    # ĐẢM BẢO DATAFRAME ĐẦY ĐỦ CỘT
    # ===============================
    def _normalize_dataframe(self, df):
        """Đảm bảo DataFrame có đủ các cột cần thiết."""

        rename_map = {
            'src_ip': 'source_ip',
            'sourceAddress': 'source_ip',
            'txPackets': 'tx_packets',
            'rxPackets': 'rx_packets',
            'txBytes': 'tx_bytes',
            'rxBytes': 'rx_bytes',
            'delaySum': 'delay_sum',
            'jitterSum': 'jitter_sum',
            'lostPackets': 'lost_packets',
            'packetLossRatio': 'packet_loss_ratio',
            'flowDuration': 'flow_duration'
        }
        df = df.rename(columns=rename_map)

        # Tạo cột nếu thiếu — tránh lỗi KeyError
        for col in ['source_ip'] + FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0

        # Fix NaN
        df = df.fillna(0)

        return df

    # ===============================
    # XỬ LÝ FLOW MỚI
    # ===============================
    def _process_new_flows(self, new_flows_df):
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
                    print(f"🚨 PHÁT HIỆN TẤN CÔNG từ IP: {ip}")
                    f.write(f"{ip}\n")
                    self.blocked_ips.add(ip)

    # ===============================
    # VÒNG LẶP CHÍNH
    # ===============================
    def watch(self):
        print(f"DEBUG: Watching file: {self.stats_file}")

        # Chờ file xuất hiện
        while not os.path.exists(self.stats_file):
            print(f"DEBUG: Đang chờ file {self.stats_file} ...")
            time.sleep(1)

        print("DEBUG: File đã xuất hiện, bắt đầu đọc realtime.")

        last_size = 0

        while True:
            try:
                # Kiểm tra thay đổi kích thước file
                current_size = os.path.getsize(self.stats_file)
                if current_size == last_size:
                    time.sleep(0.5)
                    continue

                last_size = current_size

                # Đọc file an toàn
                df = pd.read_csv(self.stats_file, on_bad_lines='skip')

                if df.empty:
                    time.sleep(0.5)
                    continue

                df = self._normalize_dataframe(df)

                # ID flow duy nhất
                df['flow_id'] = df['source_ip'] + "-" + df['tx_packets'].astype(str)

                new_flows_df = df[~df['flow_id'].isin(self.last_known_flows)].copy()

                if not new_flows_df.empty:
                    print(f"DEBUG: {len(new_flows_df)} flow mới, đang phân tích…")
                    self._process_new_flows(new_flows_df)
                    self.last_known_flows.update(new_flows_df['flow_id'])

                time.sleep(0.3)

            except pd.errors.EmptyDataError:
                time.sleep(0.2)

            except Exception as e:
                print(f"❌ Lỗi trong vòng lặp watch: {e}")
                time.sleep(1)


# Main
if __name__ == "__main__":
    mitigator = RealTimeMitigator(
        model_path=MODEL_PATH,
        stats_file=LIVE_STATS_FILE,
        blacklist_file=BLACKLIST_FILE
    )
    mitigator.watch()
