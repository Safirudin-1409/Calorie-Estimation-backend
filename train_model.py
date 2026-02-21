"""
Training script for Food Recognition Model
Trains MobileNetV2 on Food-101 dataset using transfer learning

Usage:
    python train_model.py

Requirements:
    - Food-101 dataset (auto-downloads if not present)
    - 8GB+ RAM
    - 10GB disk space
    - Optional: NVIDIA GPU with CUDA for faster training
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10  # Increase to 20-30 for better accuracy
LEARNING_RATE = 0.001
NUM_CLASSES = 101

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "food-101")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "food_model.h5")

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("🍔 Food Recognition Model Training")
print("=" * 60)
print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print("=" * 60)

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU detected: {len(gpus)} GPU(s) available")
    print(f"   Using GPU for training (10x faster!)")
else:
    print("⚠️  No GPU detected - training will use CPU (slower)")
print("=" * 60)

def download_food101():
    """Download Food-101 dataset if not present"""
    if not os.path.exists(DATA_DIR):
        print("\n📥 Downloading Food-101 dataset...")
        print("   This may take 10-15 minutes (5GB download)")
        
        # Use TensorFlow Datasets for easier download
        import tensorflow_datasets as tfds
        
        # Download and prepare dataset
        dataset, info = tfds.load('food101', 
                                  data_dir=DATA_DIR,
                                  with_info=True,
                                  as_supervised=True)
        
        print("✅ Dataset downloaded successfully!")
        return dataset, info
    else:
        print("✅ Food-101 dataset found!")
        return None, None

def create_data_generators():
    """Create data generators for training and validation"""
    print("\n📊 Preparing data generators...")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest',
        validation_split=0.2  # 80% train, 20% validation
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Check for different possible directory structures
    possible_paths = [
        os.path.join(DATA_DIR, "images"),  # food-101/images
        os.path.join(DATA_DIR, "food-101", "images"),  # food-101/food-101/images
    ]
    
    images_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            images_dir = path
            print(f"   Found images at: {path}")
            break
    
    if images_dir and os.path.exists(images_dir):
        print("   Using directory-based data loading...")
        
        train_generator = train_datagen.flow_from_directory(
            images_dir,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )
        
        val_generator = val_datagen.flow_from_directory(
            images_dir,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation'
        )
        
        return train_generator, val_generator
    else:
        print("   Could not find images directory")
        print(f"   Searched in:")
        for path in possible_paths:
            print(f"     - {path}")
        return None, None

def create_model():
    """Create MobileNetV2 model with transfer learning"""
    print("\n🏗️  Building model architecture...")
    
    # Load pre-trained MobileNetV2 (without top layer)
    base_model = MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers (transfer learning)
    base_model.trainable = False
    
    # Add custom classification head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    print("✅ Model created successfully!")
    print(f"   Total parameters: {model.count_params():,}")
    print(f"   Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    
    return model

def train_model(model, train_gen, val_gen):
    """Train the model"""
    print("\n🚀 Starting training...")
    print("=" * 60)
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    
    return history

def save_training_info(history):
    """Save training history and model info"""
    print("\n💾 Saving training information...")
    
    # Save history
    history_path = os.path.join(MODEL_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump({
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }, f, indent=2)
    
    # Save model info
    info_path = os.path.join(MODEL_DIR, "model_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Food Recognition Model\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Architecture: MobileNetV2\n")
        f.write(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}\n")
        f.write(f"Classes: {NUM_CLASSES}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Final Accuracy: {history.history['accuracy'][-1]:.4f}\n")
        f.write(f"Final Val Accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
    
    print(f"✅ Training history saved to: {history_path}")
    print(f"✅ Model info saved to: {info_path}")

def plot_training_history(history):
    """Plot training history"""
    print("\n📊 Generating training plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "training_plot.png")
    plt.savefig(plot_path)
    print(f"✅ Training plot saved to: {plot_path}")

def main():
    """Main training pipeline"""
    try:
        # Step 1: Download dataset if needed
        download_food101()
        
        # Step 2: Create data generators
        train_gen, val_gen = create_data_generators()
        
        if train_gen is None:
            print("\n❌ Error: Could not create data generators")
            print("   Please download Food-101 dataset manually:")
            print("   1. Download: http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz")
            print("   2. Extract to: backend/food-101/")
            print("   3. Run this script again")
            return
        
        # Step 3: Create model
        model = create_model()
        
        # Step 4: Train model
        history = train_model(model, train_gen, val_gen)
        
        # Step 5: Save training info
        save_training_info(history)
        
        # Step 6: Plot training history
        plot_training_history(history)
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 Training Complete!")
        print("=" * 60)
        print(f"✅ Model saved to: {MODEL_PATH}")
        print(f"✅ Final Training Accuracy: {history.history['accuracy'][-1]:.2%}")
        print(f"✅ Final Validation Accuracy: {history.history['val_accuracy'][-1]:.2%}")
        print("\n📝 Next Steps:")
        print("   1. Restart the backend server")
        print("   2. Test with real food images")
        print("   3. Enjoy high-accuracy predictions!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        print("\nTroubleshooting:")
        print("   1. Ensure you have enough disk space (10GB)")
        print("   2. Check if Food-101 dataset is properly downloaded")
        print("   3. Try reducing BATCH_SIZE if out of memory")
        print("   4. Install missing dependencies: pip install tensorflow-datasets matplotlib")

if __name__ == "__main__":
    main()
