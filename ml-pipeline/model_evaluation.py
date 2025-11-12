import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
        self.y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    def comprehensive_evaluation(self):
        """Đánh giá toàn diện model"""
        print("📊 Comprehensive Model Evaluation")
        print("=" * 50)
        
        # Basic metrics
        from sklearn.metrics import classification_report, confusion_matrix
        print("Classification Report:")
        print(classification_report(self.y_test, self.y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        self.plot_confusion_matrix(cm)
        
        # ROC Curve
        self.plot_roc_curve()
        
        # Precision-Recall Curve
        self.plot_precision_recall_curve()
        
        # Feature Importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            self.plot_feature_importance()
    
    def plot_confusion_matrix(self, cm):
        """Vẽ confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('../results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self):
        """Vẽ ROC curve"""
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('../results/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self):
        """Vẽ Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.savefig('../results/precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.show()