import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import precision_recall_curve, roc_curve, auc, classification_report, confusion_matrix

class ModelEvaluator:
    def __init__(self, model, X_test, y_test, feature_names=None):
        """
        Khởi tạo bộ đánh giá.
        :param feature_names: Danh sách tên các đặc trưng (để vẽ biểu đồ Feature Importance)
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.save_dir = '../results'
        
        # Tự động tạo thư mục lưu kết quả
        os.makedirs(self.save_dir, exist_ok=True)

        # 1. Dự đoán nhãn (Label)
        self.y_pred = model.predict(X_test)

        # 2. Dự đoán xác suất (Probability) - Xử lý linh hoạt cho SVM và RF
        if hasattr(model, "predict_proba"):
            # Random Forest, Decision Tree, etc.
            self.y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            # SVM (LinearSVC), Gradient Boosting (đôi khi)
            # ROC curve có thể làm việc với decision_function score
            self.y_pred_proba = model.decision_function(X_test)
        else:
            self.y_pred_proba = None # Không thể vẽ ROC/PR
            
    def comprehensive_evaluation(self):
        """Đánh giá toàn diện model"""
        print("\n📊 Comprehensive Model Evaluation")
        print("=" * 50)
        
        # 1. Classification Report
        print("Classification Report:")
        print(classification_report(self.y_test, self.y_pred))
        
        # 2. Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        self.plot_confusion_matrix(cm)
        
        # 3. ROC & PR Curves (Chỉ vẽ nếu có xác suất/điểm số)
        if self.y_pred_proba is not None:
            self.plot_roc_curve()
            self.plot_precision_recall_curve()
        
        # 4. Feature Importance (Hỗ trợ cả Tree và Linear models)
        self.plot_feature_importance()

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {type(self.model).__name__}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'), dpi=300)
        plt.show()
        plt.close()

    def plot_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {type(self.model).__name__}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'roc_curve.png'), dpi=300)
        plt.show()
        plt.close()

    def plot_precision_recall_curve(self):
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {type(self.model).__name__}')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'precision_recall_curve.png'), dpi=300)
        plt.show()
        plt.close()

    def plot_feature_importance(self):
        """Vẽ Feature Importance (Hỗ trợ cả Random Forest và SVM)"""
        importances = None
        
        # Trường hợp 1: Các mô hình cây (Random Forest, Decision Tree)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
        # Trường hợp 2: Các mô hình tuyến tính (LinearSVC, Logistic Regression)
        elif hasattr(self.model, 'coef_'):
            # Lấy giá trị tuyệt đối của hệ số để đánh giá mức độ ảnh hưởng
            importances = np.abs(self.model.coef_[0])
            
        if importances is None:
            print("ℹ️ Model này không hỗ trợ Feature Importance.")
            return

        # Nếu không có tên đặc trưng, tạo tên giả (Feature 0, Feature 1...)
        if self.feature_names is None:
            self.feature_names = [f"Feature {i}" for i in range(len(importances))]
            
        # Sắp xếp và vẽ
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importance - {type(self.model).__name__}")
        
        # Chỉ vẽ top 15 đặc trưng quan trọng nhất để biểu đồ không bị rối
        top_n = 15
        plt.bar(range(min(top_n, len(importances))), importances[indices][:top_n], align="center")
        plt.xticks(range(min(top_n, len(importances))), 
                   [self.feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'feature_importance.png'), dpi=300)
        plt.show()
        plt.close()