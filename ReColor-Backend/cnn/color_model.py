"""
TensorFlow-based color recognition model for ReColor colorblind detection system.
Implements a lightweight CNN for real-time color classification with CPU optimization.
"""

import tensorflow as tf
import numpy as np
import logging
from typing import Tuple, Dict, Optional
import os
import time
from utils import setup_cpu, clamp_rgb, get_color_name
from error_handler import (
    error_context, AnomalyType, AnomalyEvent, AnomalyDetector
)


class ColorModel(tf.Module):
    """
    TensorFlow-based CNN model for color recognition.
    Designed for lightweight, real-time color classification with CPU support.
    """
    
    def __init__(self, num_classes: int = 9, input_shape: Tuple[int, int, int] = (64, 64, 3)):
        """
        Initialize ColorModel.
        
        Args:
            num_classes: Number of color classes to predict
            input_shape: Input image shape (height, width, channels)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.input_shape = input_shape
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Setup enhanced error handling
        self.anomaly_detector = AnomalyDetector(self.logger)
        
        # Setup CPU
        self.cpu_info = setup_cpu()
        self.device = self.cpu_info['device']
        
        # Color class names (9 primary colors)
        self.color_classes = [
            'Red', 'Green', 'Blue', 'Yellow', 'Orange', 
            'Purple', 'Pink', 'Brown', 'Gray'
        ]
        
        # Build model
        self.model = self._build_model()
        
        # Compile model
        self._compile_model()
        
        # Model state
        self.is_trained = False
        
        self.logger.info(f"ColorModel initialized with {num_classes} classes")
        self.logger.info(f"Input shape: {input_shape}")
        self.logger.info(f"Device: {self.device}")
    
    def _build_model(self) -> tf.keras.Model:
        """
        Build lightweight CNN architecture for color classification.
        
        Returns:
            Compiled TensorFlow model
        """
        with tf.device(self.device):
            model = tf.keras.Sequential([
                # Input layer
                tf.keras.layers.Input(shape=self.input_shape),
                
                # First convolutional block
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                # Second convolutional block
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                # Third convolutional block
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                # Global average pooling instead of flatten for efficiency
                tf.keras.layers.GlobalAveragePooling2D(),
                
                # Dense layers
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                
                # Output layer
                tf.keras.layers.Dense(self.num_classes, activation='softmax', name='predictions')
            ])
        
        return model
    
    def _compile_model(self) -> None:
        """Compile the model with optimizer and loss function."""
        with tf.device(self.device):
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
    
    def predict_color(self, image: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Predict color from input image with enhanced error handling.
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB format
            
        Returns:
            Tuple of (predicted_color_name, confidence, probabilities)
        """
        start_time = time.time()
        
        with error_context("color_prediction", self.logger, self.anomaly_detector):
            
            # Validate input image
            if image is None:
                self.logger.error("Input image is None")
                anomaly = AnomalyEvent(
                    timestamp=time.time(),
                    anomaly_type=AnomalyType.INVALID_INPUT,
                    severity='high',
                    component='color_model',
                    description='Input image is None',
                    context={'input_type': type(image)}
                )
                self.anomaly_detector.log_anomaly(anomaly)
                default_probs = np.ones(self.num_classes) / self.num_classes
                return "Unknown", 0.0, default_probs
            
            if len(image.shape) != 3 or image.shape[2] != 3:
                self.logger.error(f"Invalid image shape: {image.shape}")
                anomaly = AnomalyEvent(
                    timestamp=time.time(),
                    anomaly_type=AnomalyType.INVALID_INPUT,
                    severity='high',
                    component='color_model',
                    description=f'Invalid image dimensions: {image.shape}',
                    context={'expected_dims': 3, 'actual_shape': image.shape}
                )
                self.anomaly_detector.log_anomaly(anomaly)
                default_probs = np.ones(self.num_classes) / self.num_classes
                return "Unknown", 0.0, default_probs
            
            try:
                with tf.device(self.device):
                    # Preprocess image
                    processed_image = self._preprocess_image(image)
                    
                    if processed_image is None:
                        raise ValueError("Image preprocessing failed")
                    
                    # Make prediction
                    predictions = self.model(processed_image, training=False)
                    
                    if predictions is None:
                        raise ValueError("Model prediction returned None")
                    
                    probabilities = predictions.numpy()[0]
                    
                    # Validate predictions
                    if np.any(np.isnan(probabilities)) or np.any(np.isinf(probabilities)):
                        self.logger.warning("Model returned invalid probabilities (NaN/Inf)")
                        anomaly = AnomalyEvent(
                            timestamp=time.time(),
                            anomaly_type=AnomalyType.MODEL_ERROR,
                            severity='medium',
                            component='color_model',
                            description='Model returned NaN or Inf probabilities',
                            context={'has_nan': np.any(np.isnan(probabilities)),
                                   'has_inf': np.any(np.isinf(probabilities))}
                        )
                        self.anomaly_detector.log_anomaly(anomaly)
                        # Use uniform distribution as fallback
                        probabilities = np.ones(self.num_classes) / self.num_classes
                    
                    # Get predicted class
                    predicted_class_idx = np.argmax(probabilities)
                    confidence = float(probabilities[predicted_class_idx])
                    
                    # Validate confidence
                    if confidence < 0.1:  # Very low confidence threshold
                        self.logger.warning(f"Very low prediction confidence: {confidence:.3f}")
                        anomaly = AnomalyEvent(
                            timestamp=time.time(),
                            anomaly_type=AnomalyType.MODEL_ERROR,
                            severity='low',
                            component='color_model',
                            description=f'Low prediction confidence: {confidence:.3f}',
                            context={'confidence': confidence, 'predicted_class': predicted_class_idx}
                        )
                        self.anomaly_detector.log_anomaly(anomaly)
                    
                    # Get color name
                    if predicted_class_idx < len(self.color_classes):
                        predicted_color = self.color_classes[predicted_class_idx]
                    else:
                        self.logger.error(f"Invalid predicted class index: {predicted_class_idx}")
                        predicted_color = "Unknown"
                        
                        anomaly = AnomalyEvent(
                            timestamp=time.time(),
                            anomaly_type=AnomalyType.MODEL_ERROR,
                            severity='medium',
                            component='color_model',
                            description=f'Invalid class index: {predicted_class_idx}',
                            context={'class_idx': predicted_class_idx, 'num_classes': len(self.color_classes)}
                        )
                        self.anomaly_detector.log_anomaly(anomaly)
                    
                    # Check prediction time
                    prediction_time = time.time() - start_time
                    if prediction_time > 0.05:  # 50ms threshold
                        anomaly = AnomalyEvent(
                            timestamp=time.time(),
                            anomaly_type=AnomalyType.PROCESSING_TIMEOUT,
                            severity='low',
                            component='color_model',
                            description=f'Slow prediction: {prediction_time:.3f}s',
                            context={'prediction_time': prediction_time}
                        )
                        self.anomaly_detector.log_anomaly(anomaly)
                    
                    return predicted_color, confidence, probabilities
                    
            except tf.errors.ResourceExhaustedError as e:
                self.logger.error(f"TensorFlow resource exhausted: {e}")
                anomaly = AnomalyEvent(
                    timestamp=time.time(),
                    anomaly_type=AnomalyType.RESOURCE_EXHAUSTION,
                    severity='high',
                    component='color_model',
                    description=f'TensorFlow resource exhausted: {str(e)}',
                    context={'error_type': 'resource_exhausted'}
                )
                self.anomaly_detector.log_anomaly(anomaly)
                
            except Exception as e:
                self.logger.error(f"Unexpected error in color prediction: {e}")
                anomaly = AnomalyEvent(
                    timestamp=time.time(),
                    anomaly_type=AnomalyType.MODEL_ERROR,
                    severity='medium',
                    component='color_model',
                    description=f'Color prediction failed: {str(e)}',
                    context={'error_type': type(e).__name__}
                )
                self.anomaly_detector.log_anomaly(anomaly)
            
            # Return safe fallback
            default_probs = np.ones(self.num_classes) / self.num_classes
            return "Unknown", 0.0, default_probs
    
    def _preprocess_image(self, image: np.ndarray) -> tf.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed tensor ready for inference
        """
        try:
            # Convert to float32 and normalize to [0, 1]
            if image.dtype != np.float32:
                image = image.astype(np.float32) / 255.0
            
            # Resize to model input shape
            if image.shape[:2] != self.input_shape[:2]:
                image = tf.image.resize(image, self.input_shape[:2])
            
            # Ensure correct shape
            if len(image.shape) == 3:
                image = tf.expand_dims(image, 0)  # Add batch dimension
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            # Return dummy tensor
            return tf.zeros((1, *self.input_shape), dtype=tf.float32)
    
    def load_weights(self, weights_path: str) -> bool:
        """
        Load pre-trained model weights.
        
        Args:
            weights_path: Path to saved weights file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if os.path.exists(weights_path):
                with tf.device(self.device):
                    self.model.load_weights(weights_path)
                self.is_trained = True
                self.logger.info(f"Weights loaded from {weights_path}")
                return True
            else:
                self.logger.warning(f"Weights file not found: {weights_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading weights: {e}")
            return False
    
    def save_weights(self, weights_path: str) -> bool:
        """
        Save model weights.
        
        Args:
            weights_path: Path to save weights file
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            
            with tf.device(self.device):
                self.model.save_weights(weights_path)
            
            self.logger.info(f"Weights saved to {weights_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving weights: {e}")
            return False
    
    def train_on_synthetic_data(self, 
                               samples_per_class: int = 1000,
                               epochs: int = 50,
                               validation_split: float = 0.2) -> Dict:
        """
        Train the model on synthetic color data.
        
        Args:
            samples_per_class: Number of training samples per color class
            epochs: Number of training epochs
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training history dictionary
        """
        try:
            self.logger.info("Generating synthetic training data...")
            
            # Generate synthetic data
            X_train, y_train = self._generate_synthetic_data(samples_per_class)
            
            self.logger.info(f"Training on {len(X_train)} samples...")
            
            # Define callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            # Train model
            with tf.device(self.device):
                history = self.model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    validation_split=validation_split,
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=1
                )
            
            self.is_trained = True
            self.logger.info("Training completed successfully!")
            
            return history.history
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            return {}
    
    def _generate_synthetic_data(self, samples_per_class: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic color data for training.
        
        Args:
            samples_per_class: Number of samples per color class
            
        Returns:
            Tuple of (images, labels)
        """
        # Define base colors for each class (RGB)
        base_colors = {
            0: (255, 0, 0),     # Red
            1: (0, 255, 0),     # Green
            2: (0, 0, 255),     # Blue
            3: (255, 255, 0),   # Yellow
            4: (255, 165, 0),   # Orange
            5: (128, 0, 128),   # Purple
            6: (255, 192, 203), # Pink
            7: (139, 69, 19),   # Brown
            8: (128, 128, 128)  # Gray
        }
        
        total_samples = samples_per_class * self.num_classes
        images = np.zeros((total_samples, *self.input_shape), dtype=np.float32)
        labels = np.zeros(total_samples, dtype=np.int32)
        
        sample_idx = 0
        
        for class_idx in range(self.num_classes):
            base_color = base_colors[class_idx]
            
            for _ in range(samples_per_class):
                # Add noise to base color
                noise = np.random.normal(0, 25, 3)
                noisy_color = clamp_rgb([base_color[i] + noise[i] for i in range(3)])
                
                # Create solid color image with some texture
                image = np.full(self.input_shape, noisy_color, dtype=np.float32)
                
                # Add random texture/noise
                texture_noise = np.random.normal(0, 10, self.input_shape)
                image = np.clip(image + texture_noise, 0, 255)
                
                # Normalize to [0, 1]
                image = image / 255.0
                
                images[sample_idx] = image
                labels[sample_idx] = class_idx
                sample_idx += 1
        
        # Shuffle data
        indices = np.random.permutation(total_samples)
        images = images[indices]
        labels = labels[indices]
        
        return images, labels
    
    def get_model_summary(self) -> str:
        """
        Get model architecture summary.
        
        Returns:
            Model summary string
        """
        try:
            summary_list = []
            self.model.summary(print_fn=lambda x: summary_list.append(x))
            return '\n'.join(summary_list)
        except Exception as e:
            return f"Error getting model summary: {e}"
    
    def evaluate_on_test_data(self, test_samples: int = 500) -> Dict[str, float]:
        """
        Evaluate model on synthetic test data.
        
        Args:
            test_samples: Number of test samples to generate
            
        Returns:
            Evaluation metrics dictionary
        """
        try:
            if not self.is_trained:
                self.logger.warning("Model not trained. Training on synthetic data first...")
                self.train_on_synthetic_data()
            
            # Generate test data
            X_test, y_test = self._generate_synthetic_data(test_samples // self.num_classes)
            
            # Evaluate
            with tf.device(self.device):
                results = self.model.evaluate(X_test, y_test, verbose=0)
            
            # Format results
            metrics = {
                'test_loss': float(results[0]),
                'test_accuracy': float(results[1]) * 100  # Convert to percentage
            }
            
            self.logger.info(f"Test accuracy: {metrics['test_accuracy']:.2f}%")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            return {'test_loss': float('inf'), 'test_accuracy': 0.0}
    
    def get_device_info(self) -> Dict[str, any]:
        """Get device information."""
        return self.cpu_info.copy()


# Test the ColorModel
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing ColorModel...")
    
    try:
        # Create model
        model = ColorModel()
        
        # Print model summary
        print("\nModel Architecture:")
        print(model.get_model_summary())
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Make prediction (will be random since model is not trained)
        color_name, confidence, probabilities = model.predict_color(dummy_image)
        print(f"\nTest prediction: {color_name} (confidence: {confidence:.3f})")
        
        # Test training on small dataset
        print("\nTraining model on synthetic data...")
        history = model.train_on_synthetic_data(samples_per_class=100, epochs=5)
        
        if history:
            print(f"Final training accuracy: {history['accuracy'][-1]:.3f}")
            print(f"Final validation accuracy: {history['val_accuracy'][-1]:.3f}")
        
        # Test prediction after training
        color_name, confidence, probabilities = model.predict_color(dummy_image)
        print(f"Post-training prediction: {color_name} (confidence: {confidence:.3f})")
        
        # Evaluate model
        metrics = model.evaluate_on_test_data(100)
        print(f"Evaluation results: {metrics}")
        
        print("\nColorModel test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()