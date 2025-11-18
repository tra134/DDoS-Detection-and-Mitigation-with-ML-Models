"""
train_model.py - Final Version with Comprehensive Evaluation

Tập lệnh này tải dữ liệu, huấn luyện model, tinh chỉnh tham số,
và thực hiện đánh giá toàn diện (Confusion Matrix, ROC, PR Curve).
"""

import pandas as pd
import numpy as np
import os
import glob
import joblib
import warnings
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys

# Thêm đường dẫn hiện tại vào sys.path để import các module cùng thư mục
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Import module đánh giá và tối ưu
try:
    from optimization import WOA_SSA_Hybrid
    from model_evaluation import ModelEvaluator
except ImportError as e:
    print(f"⚠️ Cảnh báo Import: {e}")
    print("   Đang chạy chế độ cơ bản (không có Optimization/Evaluation nâng cao).")
    WOA_SSA_Hybrid = None
    ModelEvaluator = None

# Tắt các cảnh báo không quan trọng
warnings.filterwarnings('ignore')

# <<< SỬA TÊN CLASS CHO ĐÚNG >>>
class DDoSTrainer:
    def __init__(self, config=None):
        self.config = config or {}
        
        # ĐẶC TRƯNG (FEATURES) QUAN TRỌNG
        self.feature_names = [
            'protocol', 'tx_packets', 'rx_packets', 'tx_bytes', 'rx_bytes',
            'delay_sum', 'jitter_sum', 'lost_packets', 'packet_loss_ratio',
            'throughput', 'flow_duration'
        ]
        
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, class_weight='balanced'),
            'Decision Tree': DecisionTreeClassifier(max_depth=15, random_state=42, class_weight='balanced'),
            'K-Neighbors': KNeighborsClassifier(n_neighbors=5, weights='distance'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            'SVM (Linear)': LinearSVC(random_state=42, class_weight='balanced', max_iter=2000, dual=False)
        }
        self.trained_models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.best_model = None

    def load_data(self, data_dir):
        """Tải và gộp tất cả file CSV, làm sạch dữ liệu."""
        print(f"📊 Đang tải TẤT CẢ bộ dữ liệu từ: {data_dir}")

        # Tìm file khớp mẫu
        search_pattern = os.path.join(data_dir, "ns3_detailed_results_*.csv")
        csv_files = glob.glob(search_pattern)

        # Fallback nếu không tìm thấy file mẫu
        if not csv_files:
            fallback_file = os.path.join(data_dir, "ns3_detailed_results.csv")
            if os.path.exists(fallback_file):
                csv_files = [fallback_file]
            else:
                raise FileNotFoundError(f"Không tìm thấy file NS3 nào tại {data_dir}")

        print(f"Tìm thấy {len(csv_files)} file để gộp lại:")
        
        all_dataframes = []
        for f in csv_files:
            print(f"  - {os.path.basename(f)}")
            try:
                df_temp = pd.read_csv(f, skipinitialspace=True)
                all_dataframes.append(df_temp)
            except Exception as e:
                print(f"  ⚠️ Lỗi khi đọc file {f}: {e}. Bỏ qua.")

        if not all_dataframes:
             raise ValueError("Không đọc được dữ liệu nào cả.")

        df = pd.concat(all_dataframes, ignore_index=True)
        print("✅ Gộp tất cả dữ liệu thành công.")
        
        # Xóa khoảng trắng trong tên cột
        df.columns = df.columns.str.strip()

        if 'label' not in df.columns:
            raise ValueError("Không tìm thấy cột 'label' trong dữ liệu.")
            
        # Lọc X và y, điền 0 nếu thiếu cột
        for col in self.feature_names:
            if col not in df.columns:
                # print(f"⚠️ Cảnh báo: Thiếu cột '{col}'. Điền giá trị 0.")
                df[col] = 0

        X = df[self.feature_names]
        y = df['label']
        
        # Xử lý dữ liệu bẩn
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        print(f"Tổng kích thước bộ dữ liệu gộp: {df.shape}")
        print(f"Phân bố nhãn cuối cùng:\n{y.value_counts(normalize=True)}")
        
        return X, y
    
    def create_synthetic_data(self, n_samples=10000):
        """Tạo dữ liệu giả (Fallback)"""
        print("🔄 Creating synthetic data (matching NS3 columns)...")
        np.random.seed(42)
        
        X = pd.DataFrame(index=range(n_samples), columns=self.feature_names)
        y = np.zeros(n_samples)
        
        n_normal = int(n_samples * 0.8)
        
        # Normal
        X.loc[:n_normal, 'protocol'] = np.random.choice([6, 17], n_normal)
        X.loc[:n_normal, 'tx_packets'] = np.random.normal(50, 10, n_normal)
        X.loc[:n_normal, 'rx_packets'] = np.random.normal(45, 10, n_normal)
        X.loc[:n_normal, 'tx_bytes'] = X['tx_packets'] * 512
        X.loc[:n_normal, 'rx_bytes'] = X['rx_packets'] * 512
        X.loc[:n_normal, 'throughput'] = np.random.normal(100, 20, n_normal)
        X.loc[:n_normal, 'packet_loss_ratio'] = 0.05

        # Attack
        start = n_normal
        X.loc[start:, 'protocol'] = 17
        X.loc[start:, 'tx_packets'] = np.random.normal(5000, 500, n_samples - start)
        X.loc[start:, 'rx_packets'] = np.random.normal(10, 5, n_samples - start)
        X.loc[start:, 'tx_bytes'] = X['tx_packets'] * 1024
        X.loc[start:, 'rx_bytes'] = X['rx_packets'] * 1024
        X.loc[start:, 'throughput'] = np.random.normal(5000, 500, n_samples - start)
        X.loc[start:, 'packet_loss_ratio'] = 0.95
        
        y[start:] = 1
        X = X.fillna(0)
        
        return X, pd.Series(y, name='label')

    def prepare_data(self, X, y):
        """Chia và Chuẩn hóa dữ liệu"""
        print("\n🔄 Chuẩn bị dữ liệu (Split & Scale)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.scaler.fit(X_train)
        X_train_scaled = pd.DataFrame(self.scaler.transform(X_train), columns=self.feature_names)
        X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), columns=self.feature_names)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_baseline(self, X_train, X_test, y_train, y_test):
        """Huấn luyện model cơ bản"""
        print("\n1️⃣ Training Baseline Model (Default Random Forest)...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"   Baseline Accuracy: {acc:.4f}")
        return model, acc

    def run_optimization(self, X_train, y_train, X_test, y_test):
        """Chạy WOA-SSA để tối ưu hóa"""
        if WOA_SSA_Hybrid is None:
            print("⚠️ Module optimization không tồn tại. Bỏ qua.")
            return None, 0.0, None

        print("\n2️⃣ Running WOA-SSA Hybrid Optimization...")
        print("   (Quá trình này có thể mất vài phút...)")
        
        optimizer = WOA_SSA_Hybrid(population_size=10, max_iter=10) # Giảm xuống để chạy nhanh demo
        best_solution, best_fitness = optimizer.optimize(X_train, y_train)
        
        optimizer.plot_convergence()
        best_model, feature_mask = optimizer.get_optimized_model(X_train, y_train)
        
        # Đánh giá trên tập test
        X_test_opt = X_test.iloc[:, feature_mask]
        y_pred = best_model.predict(X_test_opt)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"   ✨ Optimized Accuracy: {acc:.4f}")
        return best_model, acc, feature_mask

    def evaluate_and_save(self, model, X_test, y_test, feature_mask, save_path):
        """Đánh giá chi tiết và lưu model"""
        print("\n3️⃣ Final Evaluation & Saving...")
        
        # Lọc feature
        if feature_mask is not None:
            X_test_eval = X_test.iloc[:, feature_mask]
            selected_names = np.array(self.feature_names)[feature_mask]
        else:
            X_test_eval = X_test
            selected_names = self.feature_names

        # Đánh giá
        if ModelEvaluator:
            evaluator = ModelEvaluator(model, X_test_eval, y_test, feature_names=list(selected_names))
            evaluator.comprehensive_evaluation()
        else:
            print("⚠️ Module ModelEvaluator không tồn tại. Bỏ qua vẽ biểu đồ.")
            print(classification_report(y_test, model.predict(X_test_eval)))

        # Lưu model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump({
            'model': model,
            'scaler': self.scaler,
            'feature_names': list(selected_names),
            'all_feature_names': self.feature_names,
            'timestamp': datetime.now().isoformat()
        }, save_path)
        
        print(f"\n✅ Model saved to: {save_path}")
        print(f"   Features Selected: {len(selected_names)}/{len(self.feature_names)}")

if __name__ == "__main__":
    # Cấu hình đường dẫn tuyệt đối
    BASE_DIR = '/home/traphan/ns-3-dev/ddos-project-new'
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    trainer = DDoSTrainer()
    
    try:
        # 1. Load Data
        X, y = trainer.load_data(DATA_DIR)
        
        # 2. Prepare
        X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
        
        # 3. Train Baseline
        base_model, base_acc = trainer.train_baseline(X_train, X_test, y_train, y_test)
        
        # 4. Optimize
        opt_model, opt_acc, feat_mask = trainer.run_optimization(X_train, y_train, X_test, y_test)
        
        MODEL_PATH = os.path.join(MODELS_DIR, 'ddos_model.pkl')
        
        if opt_model and opt_acc >= base_acc:
            print(f"\n🏆 WOA-SSA Model chiến thắng ({opt_acc:.4f} vs {base_acc:.4f})")
            trainer.evaluate_and_save(opt_model, X_test, y_test, feat_mask, MODEL_PATH)
        else:
            print(f"\n⚠️ Baseline Model tốt hơn hoặc tương đương ({base_acc:.4f}). Lưu Baseline.")
            full_mask = np.ones(len(trainer.feature_names), dtype=bool)
            trainer.evaluate_and_save(base_model, X_test, y_test, full_mask, MODEL_PATH)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        print("⚠️ Fallback: Training with synthetic data...")
        X, y = trainer.create_synthetic_data()
        # Nếu fallback, chạy quy trình đơn giản
        trainer.scaler.fit(X) # Fit scaler
        model, acc = trainer.train_baseline(X, X, y, y) # Train trên chính nó để test
        
        # Lưu model giả
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump({
            'model': model,
            'scaler': trainer.scaler,
            'feature_names': trainer.feature_names
        }, os.path.join(MODELS_DIR, 'ddos_model.pkl'))
        print("✅ Saved synthetic model.")
    
    print("\n--- Quy trình huấn luyện hoàn tất ---")