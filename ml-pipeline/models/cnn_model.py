import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import logging

class CNNModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.logger = logging.getLogger(__name__)
    
    def build_model(self, input_shape):
        """Build CNN model architecture"""
        self.logger.info(f"Building CNN model with input shape: {input_shape}")
        
        model = models.Sequential([
            # Reshape for 1D convolution
            layers.Reshape((input_shape[0], 1), input_shape=input_shape),
            
            # First convolutional block
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # Second convolutional block
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # Global pooling and output
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train and evaluate CNN model"""
        self.logger.info("Training CNN model...")
        
        # Build model if not already built
        if self.model is None:
            self.build_model(X_train.shape[1:])
        
        # Reshape data for CNN
        X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Get training parameters
        cnn_params = self.config.get('models', {}).get('cnn', {})
        
        # Train model
        history = self.model.fit(
            X_train_cnn, y_train,
            epochs=cnn_params.get('epochs', 10),
            batch_size=cnn_params.get('batch_size', 32),
            validation_data=(X_test_cnn, y_test),
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(X_test_cnn, y_test, verbose=0)
        
        results = {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'history': history.history,
            'model_type': 'cnn'
        }
        
        self.logger.info(f"CNN training completed - Accuracy: {test_accuracy:.4f}")
        return results