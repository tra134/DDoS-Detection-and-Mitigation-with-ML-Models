import pandas as pd
import joblib
import time
import os
import warnings
from io import StringIO

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = "/home/traphan/ns-3-dev/ddos-project-new"

MODEL_PATH = os.path.join(BASE_DIR, "models/ddos_model.pkl")
LIVE_DATA_DIR = os.path.join(BASE_DIR, "data", "live")
LIVE_STATS_FILE = os.path.join(LIVE_DATA_DIR, "live_flow_stats.csv")
BLACKLIST_FILE = os.path.join(LIVE_DATA_DIR, "blacklist.txt")

FULL_FEATURE_COLUMNS = [
    'protocol', 'tx_packets', 'rx_packets', 'tx_bytes', 'rx_bytes',
    'delay_sum', 'jitter_sum', 'lost_packets', 'packet_loss_ratio',
    'throughput', 'flow_duration'
]

HEADER_NAMES = [
    "time","source_ip","protocol","tx_packets","rx_packets","tx_bytes","rx_bytes",
    "delay_sum","jitter_sum","lost_packets","packet_loss_ratio",
    "throughput","flow_duration","label"
]


class LiveMitigator:
    def __init__(self):
        print("=== Loading ML Model ===")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ Model file NOT FOUND: {MODEL_PATH}")

        m = joblib.load(MODEL_PATH)

        self.model = m["model"]
        self.scaler = m["scaler"]
        self.selected_features = m["feature_names"]

        print(f"✅ Model loaded. Using features: {self.selected_features}")

        # File tracking
        self.last_size = 0
        self.blocked_ips = set()

        os.makedirs(LIVE_DATA_DIR, exist_ok=True)

        if os.path.exists(BLACKLIST_FILE):
            os.remove(BLACKLIST_FILE)

        print("⬤ Watching:", LIVE_STATS_FILE)
        print("⬤ Blacklist:", BLACKLIST_FILE)

    # --------------------------------------------------------------
    def normalize_dataframe(self, df):
        df = df.copy()
        df.columns = df.columns.str.strip()

        # Fill missing feature columns
        for col in FULL_FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0

        df = df.fillna(0)
        return df

    # --------------------------------------------------------------
    def process_new_flows(self, df):
        if df.empty:
            return

        df = self.normalize_dataframe(df)

        # Step 1: scale 11 full features
        X_full = df[FULL_FEATURE_COLUMNS]

        try:
            X_scaled = self.scaler.transform(X_full)
        except Exception as e:
            print("❌ Scaler error:", e)
            return

        X_scaled_df = pd.DataFrame(X_scaled, columns=FULL_FEATURE_COLUMNS)

        # Step 2: select features for ML model
        X_final = X_scaled_df[self.selected_features]

        # Step 3: predict
        preds = self.model.predict(X_final)
        df["prediction"] = preds

        attackers = df[df["prediction"] == 1]

        if attackers.empty:
            return

        for ip in attackers["source_ip"].unique():
            if ip not in self.blocked_ips:
                print(f"🚨 BLOCK: Detected attacker IP = {ip}")
                with open(BLACKLIST_FILE, "a") as f:
                    f.write(ip + "\n")
                self.blocked_ips.add(ip)

    # --------------------------------------------------------------
    def watch(self):
        print("=== Mitigation Engine Running ===")

        # Wait for file creation
        while not os.path.exists(LIVE_STATS_FILE):
            print("⏳ Waiting for NS-3 to create live_flow_stats.csv ...")
            time.sleep(1)

        print("📄 Detected live_flow_stats.csv, starting live analysis.")

        while True:
            try:
                size = os.path.getsize(LIVE_STATS_FILE)

                # file reset
                if size < self.last_size:
                    print("🔄 File reset detected. Restarting reading pointer.")
                    self.last_size = 0

                # no new data
                if size == self.last_size:
                    time.sleep(0.3)
                    continue

                # read new data
                with open(LIVE_STATS_FILE, "r") as f:
                    f.seek(self.last_size)
                    new_data = f.read()
                    self.last_size = f.tell()

                # Skip header duplicates
                lines = [l for l in new_data.splitlines() if not l.startswith("time,source_ip")]

                if not lines:
                    continue

                csv_chunk = "\n".join(lines)
                df = pd.read_csv(StringIO(csv_chunk), names=HEADER_NAMES)

                self.process_new_flows(df)

            except Exception as e:
                print("❌ Main Loop Error:", e)
                time.sleep(0.5)


if __name__ == "__main__":
    mitigator = LiveMitigator()
    mitigator.watch()
