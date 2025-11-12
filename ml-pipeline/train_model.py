"""
train_model.py

Tập lệnh này tải tất cả dữ liệu mô phỏng NS3 từ thư mục data/raw,
gộp chúng lại, sau đó huấn luyện nhiều mô hình ML để tìm ra
mô hình phát hiện DDoS tốt nhất và lưu nó lại.
"""

import pandas as pd
import numpy as np
import os
import glob  # Để tìm kiếm file
import joblib
import warnings
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC  # Nhanh hơn SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Tắt các cảnh báo không quan trọng
warnings.filterwarnings('ignore')


class DDoSTrainer:
    def __init__(self, config=None):
        self.config = config or {}
        
        # SỬA: Định nghĩa rõ ràng các đặc trưng CHỈ LÀ SỐ
        # Đây là các cột duy nhất chúng ta dùng để huấn luyện.
        self.feature_names = [
            'protocol', 'tx_packets', 'rx_packets', 'tx_bytes', 'rx_bytes',
            'delay_sum', 'jitter_sum', 'lost_packets', 'packet_loss_ratio',
            'throughput', 'flow_duration'
        ]
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=20, 
                random_state=42,
                class_weight='balanced'
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=15, 
                random_state=42,
                class_weight='balanced'
            ),
            'K-Neighbors': KNeighborsClassifier(n_neighbors=5, weights='distance'),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                random_state=42
            ),
            'SVM (Linear)': LinearSVC(random_state=42, class_weight='balanced', max_iter=2000, dual=True)
        }
        self.trained_models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.best_model = None

    def load_data(self, data_dir):
        """
        SỬA: Tải và gộp tất cả file CSV, sau đó CHỈ CHỌN
        các đặc trưng (feature_names) đã định nghĩa.
        """
        print(f"📊 Đang tải TẤT CẢ bộ dữ liệu từ: {data_dir}")

        search_pattern = os.path.join(data_dir, "ns3_detailed_results_*_nodes.csv")
        csv_files = glob.glob(search_pattern)

        if not csv_files:
            raise FileNotFoundError(f"Không tìm thấy file NS3 nào tại {search_pattern}")

        print(f"Tìm thấy {len(csv_files)} file để gộp lại:")
        all_dataframes = [pd.read_csv(file_path) for file_path in csv_files]
        df = pd.concat(all_dataframes, ignore_index=True)
        print("✅ Gộp tất cả dữ liệu thành công.")
        
        # SỬA: Lọc X để CHỈ chứa các đặc trưng (features) đã định nghĩa
        if 'label' not in df.columns:
            raise ValueError("Không tìm thấy cột 'label' trong dữ liệu.")
            
        try:
            # X (Features) chỉ bao gồm các cột trong self.feature_names
            X = df[self.feature_names] 
            y = df['label']
        except KeyError as e:
            print(f"Lỗi: Không tìm thấy các đặc trưng cần thiết trong file CSV. Thiếu: {e}")
            raise
        
        # Xử lý các giá trị không hợp lệ (ví dụ: inf)
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        print(f"Tổng kích thước bộ dữ liệu gộp: {df.shape}")
        print(f"Số lượng đặc trưng đã chọn: {X.shape[1]}")
        print(f"Phân bố nhãn cuối cùng:\n{y.value_counts(normalize=True)}")
        
        return X, y
    
    def create_synthetic_data(self, n_samples=10000):
        """
        SỬA: Tạo dữ liệu giả khớp với các đặc trưng của NS3.
        """
        print("🔄 Creating synthetic data (matching NS3 columns)...")
        np.random.seed(42)
        
        X = pd.DataFrame(index=range(n_samples), columns=self.feature_names)
        y = np.zeros(n_samples)
        
        # Tạo dữ liệu normal (80%)
        n_normal = int(n_samples * 0.8)
        X.loc[:n_normal, 'protocol'] = np.random.choice([6, 17], n_normal) # TCP/UDP
        X.loc[:n_normal, 'tx_packets'] = np.random.normal(50, 10, n_normal)
        X.loc[:n_normal, 'rx_packets'] = np.random.normal(45, 10, n_normal)
        X.loc[:n_normal, 'tx_bytes'] = X['tx_packets'] * 512
        X.loc[:n_normal, 'rx_bytes'] = X['rx_packets'] * 512
        X.loc[:n_normal, 'delay_sum'] = np.random.normal(0.5, 0.1, n_normal)
        X.loc[:n_normal, 'jitter_sum'] = np.random.normal(0.1, 0.05, n_normal)
        X.loc[:n_normal, 'lost_packets'] = np.random.randint(0, 5, n_normal)
        X.loc[:n_normal, 'packet_loss_ratio'] = X['lost_packets'] / (X['tx_packets'] + 1)
        X.loc[:n_normal, 'flow_duration'] = np.random.normal(10, 2, n_normal)
        X.loc[:n_normal, 'throughput'] = (X['rx_bytes'] * 8) / (X['flow_duration'] * 1000 + 1) # Kbps

        # Tạo dữ liệu attack (20%)
        n_attack = n_samples - n_normal
        start_index = n_normal
        X.loc[start_index:, 'protocol'] = 17 # UDP
        X.loc[start_index:, 'tx_packets'] = np.random.normal(5000, 500, n_attack)
        X.loc[start_index:, 'rx_packets'] = np.random.normal(10, 5, n_attack) # Bị server drop
        X.loc[start_index:, 'tx_bytes'] = X['tx_packets'] * 1024
        X.loc[start_index:, 'rx_bytes'] = X['rx_packets'] * 1024
        X.loc[start_index:, 'delay_sum'] = np.random.normal(2.0, 0.5, n_attack) # Delay cao
        X.loc[start_index:, 'jitter_sum'] = np.random.normal(1.0, 0.2, n_attack)
        X.loc[start_index:, 'lost_packets'] = np.random.normal(4900, 500, n_attack)
        X.loc[start_index:, 'packet_loss_ratio'] = X['lost_packets'] / (X['tx_packets'] + 1)
        X.loc[start_index:, 'flow_duration'] = np.random.normal(5, 1, n_attack) # Ngắn
        X.loc[start_index:, 'throughput'] = (X['rx_bytes'] * 8) / (X['flow_duration'] * 1000 + 1)
        
        y[start_index:] = 1  # Attack labels
        
        X = X.fillna(0)
        y_series = pd.Series(y, name='label')
        
        print(f"Synthetic dataset created: {X.shape}")
        return X, y_series
    
    def train_models(self, X, y):
        """
        SỬA: X_train đã là DataFrame chỉ chứa các cột số.
        Không cần X_train[self.feature_names] nữa.
        """
        # 1. Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 2. SỬA: Scaling (Fit trên X_train, transform trên cả hai)
        print("\n🔄 Scaling data (Fit on train, transform train/test)...")
        self.scaler.fit(X_train) # X_train đã là DataFrame chỉ chứa các đặc trưng số
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("\n🎯 Training models...")
        results = {}
        
        for name, model in self.models.items():
            print(f"\n📈 Training {name}...")
            model.fit(X_train_scaled, y_train) 
            self.trained_models[name] = model
            
            y_pred = model.predict(X_test_scaled)
            
            # Tính toán AUC
            try:
                if hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else: 
                    decision_values = model.decision_function(X_test_scaled)
                    y_pred_proba = (decision_values - decision_values.min()) / (decision_values.max() - decision_values.min())
            except Exception:
                y_pred_proba = y_pred
            
            accuracy = accuracy_score(y_test, y_pred)
            if len(np.unique(y_test)) > 1:
                auc_score = roc_auc_score(y_test, y_pred_proba)
            else:
                auc_score = 0.0
                
            results[name] = {'accuracy': accuracy, 'auc_score': auc_score, 'model': model}
            
            print(f"✅ {name} Results: Accuracy: {accuracy:.4f}, AUC Score: {auc_score:.4f}")
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                self.feature_importance[name] = np.abs(model.coef_[0])
        
        # Chọn model tốt nhất
        best_model_name = max(results, key=lambda x: results[x]['auc_score'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\n🏆 Best Model: {best_model_name}")
        
        # Hiển thị detailed report
        y_pred_best = self.best_model.predict(X_test_scaled)
        print(f"\n📊 Detailed Report for {best_model_name}:")
        print(classification_report(y_test, y_pred_best))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
        
        return results, self.best_model, X_train_scaled, X_test_scaled, y_train, y_test
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Tinh chỉnh trên X_train đã được scale"""
        print("\n🔧 Performing hyperparameter tuning...")
        
        best_model_name = type(self.best_model).__name__
        param_grid = {}
        model_to_tune = None
        
        if 'RandomForest' in best_model_name:
            param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
            model_to_tune = RandomForestClassifier(random_state=42, class_weight='balanced')
        elif 'GradientBoosting' in best_model_name:
            param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05]}
            model_to_tune = GradientBoostingClassifier(random_state=42)
        else:
            print(f"No parameter grid defined for {best_model_name}. Skipping tuning.")
            return self.best_model

        grid_search = GridSearchCV(model_to_tune, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train) # Dùng X_train đã scale
        
        print(f"Best parameters: {grid_search.best_params_}")
        self.best_model = grid_search.best_estimator_
        return self.best_model
    
    def plot_feature_importance(self, save_path=None):
        """Vẽ biểu đồ feature importance"""
        if not self.feature_importance:
            print("No feature importance data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        plot_count = 0
        for idx, (name, importance) in enumerate(list(self.feature_importance.items())):
            if plot_count >= 4: break
            if importance is None: continue
                
            indices = np.argsort(importance)[::-1][:10] # Top 10
            axes[plot_count].barh(range(len(indices)), importance[indices])
            axes[plot_count].set_yticks(range(len(indices)))
            axes[plot_count].set_yticklabels([self.feature_names[i] for i in indices])
            axes[plot_count].set_title(f'Feature Importance - {name}')
            axes[plot_count].set_xlabel('Importance')
            plot_count += 1
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        plt.show()
    
    def save_model(self, model, file_path):
        """SỬA: Tự tạo thư mục"""
        model_directory = os.path.dirname(file_path)
        os.makedirs(model_directory, exist_ok=True)
        
        joblib.dump({
            'model': model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_time': datetime.now().isoformat(),
            'config': self.config
        }, file_path)
        
        print(f"✅ Model saved to {file_path}")
    
    def train_with_sample_data(self, save_path):
        """Train với dữ liệu mẫu (fallback)"""
        X, y = self.create_synthetic_data(10000)
        results, best_model, _, _, _, _ = self.train_models(X, y)
        self.save_model(best_model, save_path)
        return results, best_model

# --- HÀM CHẠY CHÍNH (MAIN) ---
if __name__ == "__main__":
    
    # SỬA: Dùng đường dẫn tuyệt đối
    BASE_DIR = '/home/traphan/ns-3-dev/ddos-project-new'
    
    CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'ml-config.yaml')
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw') 
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'ddos_model.pkl')
    FEATURES_SAVE_PATH = os.path.join(RESULTS_DIR, 'feature_importance.png')
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
    except Exception:
        config = {}
    
    trainer = DDoSTrainer(config)
    
    try:
        print(f"--- Bắt đầu quy trình huấn luyện ---")
        X, y = trainer.load_data(DATA_DIR) 
        
        results, best_model, X_train_s, X_test_s, y_train, y_test = trainer.train_models(X, y)
        
        best_model = trainer.hyperparameter_tuning(X_train_s, y_train)
        
        trainer.plot_feature_importance(FEATURES_SAVE_PATH)
        
        trainer.save_model(best_model, MODEL_SAVE_PATH)
        
    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file dữ liệu NS3 nào trong {DATA_DIR}.")
        print("Chuyển sang dùng dữ liệu synthetic (dữ liệu giả)...")
        trainer.train_with_sample_data(MODEL_SAVE_PATH)
    
    print("\n--- Quy trình huấn luyện hoàn tất ---")