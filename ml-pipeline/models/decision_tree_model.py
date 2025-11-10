import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

class DecisionTreeModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.logger = logging.getLogger(__name__)
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train and evaluate Decision Tree model"""
        self.logger.info("Training Decision Tree model...")
        
        # Get model parameters from config
        model_params = self.config.get('models', {}).get('decision_tree', {})
        
        # Create and train model
        self.model = DecisionTreeClassifier(
            max_depth=model_params.get('max_depth', 10),
            min_samples_split=model_params.get('min_samples_split', 2),
            min_samples_leaf=model_params.get('min_samples_leaf', 1),
            random_state=model_params.get('random_state', 42)
        )
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'predictions': y_pred.tolist(),
            'feature_importance': self.model.feature_importances_.tolist(),
            'model_type': 'decision_tree'
        }
        
        self.logger.info(f"Decision Tree training completed - Accuracy: {accuracy:.4f}")
        return results