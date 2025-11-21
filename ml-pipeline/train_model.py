import pandas as pd
import numpy as np
import os
import glob
import joblib
import warnings
import sys

# --- MATPLOTLIB CONFIG ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# -------------------------

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.getcwd()

sys.path.append(CURRENT_DIR)

try:
    from optimization import WOA_SSA_Hybrid
    from model_evaluation import ModelEvaluator
except ImportError:
    WOA_SSA_Hybrid = None
    ModelEvaluator = None

warnings.filterwarnings('ignore')


BASE_DIR = '/home/traphan/ns-3-dev/ddos-project-new'
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


class DDoSTrainer:
    def __init__(self):
        self.all_feature_names = [
            'protocol', 'tx_packets', 'rx_packets', 'tx_bytes', 'rx_bytes',
            'delay_sum', 'jitter_sum', 'lost_packets', 'packet_loss_ratio',
            'throughput', 'flow_duration'
        ]
        self.scaler = StandardScaler()

    # ----------------------------------------------------
    #               LOAD & CLEAN DATA

    def load_data(self):
        print(f"Loading data from: {DATA_DIR}")

        files = glob.glob(os.path.join(DATA_DIR, "ns3_detailed_results*.csv"))
        if not files:
            raise FileNotFoundError(" Do not find CSV file in folder raw/ !")

        dfs = []
        for f in files:
            try:
                df_temp = pd.read_csv(f, skipinitialspace=True)
                dfs.append(df_temp)
            except Exception as e:
                print(f" Warning: Cannot read {f} – {e}")

        if not dfs:
            raise ValueError(" Loaded file failed – inappropriate dataset.")

        df = pd.concat(dfs, ignore_index=True)
        df.columns = df.columns.str.strip()
        df = df.fillna(0)

        # Đảm bảo đủ cột
        for col in self.all_feature_names:
            if col not in df.columns:
                print(f" Missing column '{col}', filling with 0.")
                df[col] = 0

        print(f"Loaded {len(df)} records.")

        if 'label' not in df.columns:
            raise ValueError(" lack of 'label' collum in dataset!")

        return df[self.all_feature_names], df['label']

    # ----------------------------------------------------
    #                     TRAINER
    # ----------------------------------------------------
    def run(self):
        # ------------------ Load ------------------
        X, y = self.load_data()

        # ------------------ Split -----------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # ------------------ Scale -----------------
        print("\n Scaling data...")
        self.scaler.fit(X_train)

        X_train_scaled = pd.DataFrame(
            self.scaler.transform(X_train), columns=self.all_feature_names
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test), columns=self.all_feature_names
        )

        # ----------------------------------------------------
        #          WOA + SSA OPTIMIZATION IF AVAILABLE

        if WOA_SSA_Hybrid is not None:
            print("\n Running WOA-SSA Optimization...")

            optimizer = WOA_SSA_Hybrid(population_size=10, max_iter=10)
            try:
                best_model, feature_mask = optimizer.get_optimized_model(
                    X_train_scaled, y_train
                )
            except Exception as e:
                print(f"Optimizer crashed: {e}")
                print("Switching to default RandomForest.")
                best_model = RandomForestClassifier(n_estimators=100)
                best_model.fit(X_train_scaled, y_train)
                feature_mask = np.array([True] * len(self.all_feature_names))

            # Plot convergence if possible
            try:
                optimizer.plot_convergence()
            except Exception:
                pass

            selected_features = np.array(self.all_feature_names)[feature_mask]
            X_test_final = X_test_scaled.iloc[:, feature_mask]

            print(f"Selected Features: {list(selected_features)}")

        # ----------------------------------------------------
        #         DEFAULT RANDOM FOREST IF OPTIMIZER FAILS

        else:
            print("Optimizer not found → Using Default Random Forest")
            best_model = RandomForestClassifier(n_estimators=100)
            best_model.fit(X_train_scaled, y_train)
            selected_features = self.all_feature_names
            X_test_final = X_test_scaled

        # ----------------------------------------------------

        print("\nEvaluating model...")

        if ModelEvaluator is not None:
            evaluator = ModelEvaluator(
                best_model,
                X_test_final,
                y_test,
                feature_names=list(selected_features),
                save_dir=RESULTS_DIR
            )
            evaluator.comprehensive_evaluation()
        else:
            print(classification_report(y_test, best_model.predict(X_test_final)))

        # ----------------------------------------------------
           
        saved_path = os.path.join(MODELS_DIR, 'ddos_model.pkl')

        joblib.dump({
            'model': best_model,
            'scaler': self.scaler,
            'feature_names': list(selected_features),
            'all_feature_names': self.all_feature_names
        }, saved_path)

        print(f"\n Model saved to: {saved_path}")


# ----------------------------------------------------
#                    MAIN
if __name__ == "__main__":
    trainer = DDoSTrainer()
    trainer.run()
