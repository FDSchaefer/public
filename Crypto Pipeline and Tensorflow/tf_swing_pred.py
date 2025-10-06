"""
TensorFlow LSTM Batch Trainer for Memory-Constrained Systems
Uses 12-hour sequences (720 periods at 1-minute intervals)

Prerequisites:
- CUDA-enabled GPU
- Generated .npy files from tf_feature_engineering.py

Usage:
python tf_batch_trainer.py
"""

# Disable GPU, use CPU only
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import gc
import warnings
warnings.filterwarnings('ignore')

useGPU = False
# Enable GPU memory growth to prevent OOM
gpus = tf.config.list_physical_devices('GPU')
if useGPU and gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU available: {len(gpus)} device(s)")
else:
    print("WARNING: No GPU detected, training will be slow")
    print("Running on CPU")

tf.keras.mixed_precision.set_global_policy('mixed_float16')



class MemoryEfficientTrainer:
    """LSTM trainer with batch processing for limited memory."""
    
    def __init__(self, sequence_length=360, batch_size=16):
        """
        Args:
            sequence_length: 360 periods = 6 hours (reduced for CPU efficiency)
            batch_size: Small batch for 8GB memory and CPU
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.model = None
        self.history = None
        
    def load_data(self):
        """Load preprocessed numpy arrays."""
        print("Loading data...")
        X_train = np.load('X_train.npy')
        X_test = np.load('X_test.npy')
        y_train = np.load('y_train.npy')
        y_test = np.load('y_test.npy')
        
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def create_sequences(self, X, y):
        """Create sequences from flat data."""
        n_samples = len(X) - self.sequence_length
        n_samples = max(n_samples, 0)
        n_features = X.shape[1]
        
        # Pre-allocate array
        X_seq = np.zeros((n_samples, self.sequence_length, n_features), dtype=np.float16)
        y_seq = np.zeros(n_samples, dtype=np.float16)
        
        # Fill in batches to save memory
        print(f"Creating {n_samples} sequences...")
        chunk_size = 1000
        
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            for j in range(i, end_idx):
                X_seq[j] = X[j:j + self.sequence_length]
                y_seq[j] = y[j + self.sequence_length]
            
            if (i // chunk_size) % 5 == 0:
                print(f"  Processed {end_idx}/{n_samples} sequences")
                gc.collect()
        
        print(f"Sequence shape: {X_seq.shape}")
        return X_seq, y_seq
    
    def build_model(self, n_features):
        """Build LSTM model optimized for CPU."""
        model = models.Sequential([
            # Single LSTM layer with recurrent_dropout for CPU efficiency
            layers.LSTM(
                64, 
                return_sequences=False,
                recurrent_dropout=0.2,
                input_shape=(self.sequence_length, n_features),
                unroll=False  # Important for CPU
            ),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            
            # Output layer (binary classification)
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall()]
        )
        
        return model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=50, validation_split=0.1):
        """Train model with memory-efficient batch processing."""
        
        n_features = X_train.shape[1]
        
        # Build model
        print("\nBuilding model...")
        self.model = self.build_model(n_features)
        self.model.summary()
        
        # Create sequences
        print("\nCreating training sequences...")
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        gc.collect()
        
        print("\nCreating test sequences...")
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test)
        gc.collect()
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        checkpoint = callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # Train with batch processing
        print(f"\nTraining with batch_size={self.batch_size}...")
        print(f"This will process {len(X_train_seq) // self.batch_size} batches per epoch\n")
        
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=1
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_acc, test_prec, test_rec = self.model.evaluate(
            X_test_seq, y_test_seq,
            batch_size=self.batch_size,
            verbose=1
        )
        
        print(f"\nTest Results:")
        print(f"  Loss:      {test_loss:.4f}")
        print(f"  Accuracy:  {test_acc:.4f}")
        print(f"  Precision: {test_prec:.4f}")
        print(f"  Recall:    {test_rec:.4f}")
        
        # Cleanup
        del X_train_seq, y_train_seq, X_test_seq, y_test_seq
        gc.collect()
        
        return self.history
    
    def plot_history(self):
        """Plot training history."""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Train Acc')
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Val Acc')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train Prec')
        axes[1, 0].plot(self.history.history['val_precision'], label='Val Prec')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("Training history saved to training_history.png")
        plt.show()
    
    def save_model(self, filepath='crypto_model.keras'):
        """Save trained model."""
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='crypto_model.keras'):
        """Load saved model."""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


def main():
    # Initialize trainer
    # 360 periods = 12 hours at 5-minute intervals
    # batch_size=16 for 8GB memory on CPU
    trainer = MemoryEfficientTrainer(
        sequence_length=int(24*(60/5)),
        batch_size=32
    )
    
    try:
        # Load data
        X_train, X_test, y_train, y_test = trainer.load_data()
        
        # Train model
        trainer.train(
            X_train, y_train,
            X_test, y_test,
            epochs=50,
            validation_split=0.1
        )
        
        # Plot results
        trainer.plot_history()
        
        # Save model
        trainer.save_model('crypto_lstm.keras')
        
        print("\nTraining complete!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Force garbage collection
        gc.collect()
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()