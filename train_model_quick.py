"""
Quick training script for Food Recognition Model
Uses a smaller subset for faster training (5-10 minutes)

This is ideal for:
- Testing the training pipeline
- Quick prototyping
- Limited computational resources

For full accuracy, use train_model.py instead
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from datetime import datetime

# Configuration
IMAGE_SIZE = 224
BATCH_SIZE = 16  # Smaller batch size for faster training
EPOCHS = 5  # Fewer epochs for quick training
LEARNING_RATE = 0.001
NUM_CLASSES = 101

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "food_model.h5")

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("🚀 Quick Training Mode")
print("=" * 60)
print("This will create a functional model in 5-10 minutes")
print("For best accuracy, use train_model.py instead")
print("=" * 60)

def create_model():
    """Create MobileNetV2 model"""
    print("\n🏗️  Building model...")
    
    base_model = MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze most layers, only train top layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model created!")
    return model

def create_synthetic_data():
    """Create synthetic training data for quick testing"""
    print("\n📊 Creating synthetic training data...")
    print("   (For real training, use actual Food-101 dataset)")
    
    # Generate random images and labels
    num_samples = 1000
    X_train = np.random.rand(num_samples, IMAGE_SIZE, IMAGE_SIZE, 3).astype('float32')
    y_train = keras.utils.to_categorical(
        np.random.randint(0, NUM_CLASSES, num_samples),
        NUM_CLASSES
    )
    
    X_val = np.random.rand(200, IMAGE_SIZE, IMAGE_SIZE, 3).astype('float32')
    y_val = keras.utils.to_categorical(
        np.random.randint(0, NUM_CLASSES, 200),
        NUM_CLASSES
    )
    
    print(f"✅ Created {num_samples} training samples")
    return (X_train, y_train), (X_val, y_val)

def main():
    """Quick training pipeline"""
    print("\n⚠️  WARNING: This uses synthetic data for demonstration")
    print("   For real predictions, you need to train on Food-101 dataset")
    print("   Use train_model.py for production-ready model\n")
    
    # Create model
    model = create_model()
    
    # Create synthetic data
    (X_train, y_train), (X_val, y_val) = create_synthetic_data()
    
    # Train
    print("\n🚀 Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    # Save model
    print(f"\n💾 Saving model to: {MODEL_PATH}")
    model.save(MODEL_PATH)
    
    # Save info
    info_path = os.path.join(MODEL_DIR, "model_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Quick Training Model (Synthetic Data)\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Note: Trained on synthetic data for testing\n")
        f.write(f"For production use, train on Food-101 dataset\n")
    
    print("\n" + "=" * 60)
    print("✅ Quick training complete!")
    print("=" * 60)
    print(f"Model saved to: {MODEL_PATH}")
    print("\n⚠️  IMPORTANT:")
    print("   This model was trained on synthetic data")
    print("   It will NOT give accurate predictions")
    print("   Use train_model.py with Food-101 dataset for real accuracy")
    print("=" * 60)

if __name__ == "__main__":
    main()
