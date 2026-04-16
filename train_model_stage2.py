# train_stage2.py
"""
STAGE 2: Freshness Classifiers
Train one specialized model for EACH produce type
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import os
import random
import numpy as np
import matplotlib.pyplot as plt

print("\n STAGE 2: Ripeness Classifier\n")

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 12 
EPOCHS = 30  
BASE_DIR = "data"
SEED = 42
VAL_SPLIT = 0.2
MAX_CLASS_WEIGHT = 2.5

# Keep shuffling/initialization reproducible across runs.
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# Produce types to train
PRODUCE_TYPES = ["strawberry", "spinach", "tomato", "banana", "avocado"]

def train_freshness_model(produce_name):
    """
    Train a freshness classifier for a specific produce type
    """
    print(f"\n Training {produce_name.upper()} Freshness Classifier\n")
    
    data_dir = os.path.join(BASE_DIR, produce_name)
    
    # Check if folder exists
    if not os.path.exists(data_dir):
        print(f" Skipping {produce_name} - folder not found: {data_dir}")
        return None
    
    # Count images
    total_images = sum([len(files) for r, d, files in os.walk(data_dir)])
    print(f" Found {total_images} images for {produce_name}")
    
    # Load data
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )
    
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )
    
    class_names = train_ds.class_names
    print(f" Freshness stages for {produce_name}: {class_names}")

    # Give extra weight to rare classes so the model does not ignore them.
    class_counts = {i: 0 for i in range(len(class_names))}
    for _, labels in train_ds:
        for label in labels.numpy():
            class_counts[int(label)] += 1

    total_train = sum(class_counts.values())
    class_weights = {}
    for class_index, count in class_counts.items():
        if count == 0:
            class_weights[class_index] = 1.0
            print(f" Warning: class '{class_names[class_index]}' has 0 training images.")
            continue
        raw_weight = total_train / (len(class_names) * count)
        # Cap very large weights so tiny classes don't swing training too hard.
        class_weights[class_index] = min(raw_weight, MAX_CLASS_WEIGHT)
        if count < 10:
            print(
                f" Warning: class '{class_names[class_index]}' has only {count} training images; "
                "results may vary a lot."
            )

    print(f" Class counts for {produce_name}: {class_counts}")
    print(f" Class weights for {produce_name}: {class_weights}")
    
    # Optimize pipeline with caching and prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    # Use prefetch() to say "while the GPU is currently training on Batch 1, 
    # start preparing Batch 2 right now"
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # define eraly stopping and learning rate scheduler
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=4, 
        restore_best_weights=True
    )
    # ReduceLROnPlateau patience MUST be smaller than the EarlyStopping
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=2, 
        min_lr=1e-6
    )
    os.makedirs('models/stage2_freshness_classifiers', exist_ok=True)
    best_model = keras.callbacks.ModelCheckpoint(
        filepath=f"models/stage2_freshness_classifiers/{produce_name}_best.keras",
        monitor="val_loss",
        save_best_only=True
    )

    # Data augmentation (inline = more variations)
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal", seed=SEED),
        # layers.RandomFlip("vertical", seed=SEED),
        layers.RandomRotation(0.05),    
        # layers.RandomZoom(0.15),
        # layers.RandomContrast(0.1),
    ])
    
    # Build model
    print(f"\n Building {produce_name} freshness model...")
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    # Keep regularization moderate; too much can make tiny datasets unstable.
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(
        len(class_names), 
        activation='softmax',
        kernel_regularizer=keras.regularizers.l2(0.0001)
    )(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),  
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Training specifics 
    print(f"\n Training {produce_name} model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        verbose=1, 
        class_weight=class_weights
    )

    # Phase 2: FINE-TUNING 
    print(f"\n Phase 2: Fine-Tuning {produce_name} backbone...")
    base_model.trainable = True
    
    # loop through every single layer in the backbone
    for layer in base_model.layers:
        # If the layer is a Batch Normalization layer, freeze it!
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
            
    # also freeze the first 110 layers 
    for layer in base_model.layers[:110]:
        layer.trainable = False
        
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00001), 
        loss='sparse_categorical_crossentropy', 
        metrics=["accuracy"]
    )
                  
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=40, # Higher max epochs
        callbacks=[early_stopping, lr_scheduler, best_model], 
        verbose=1,
        class_weight=class_weights
    )
    
    # Save model
    model_path = f'models/stage2_freshness_classifiers/{produce_name}_freshness.keras'
    model.save(model_path)
    
    # Save class names
    with open(f'models/stage2_freshness_classifiers/{produce_name}_classes.txt', 'w') as f:
        f.write('\n'.join(class_names))
    
    # Plot
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{produce_name.title()} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{produce_name.title()} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'stage2_{produce_name}_training.png')
    # Close figure so memory stays stable while looping through many models.
    plt.close()
    
    # Results
    final_acc = history.history['val_accuracy'][-1]
    print(f"\n {produce_name.upper()} Model Complete!")
    print(f"   Validation Accuracy: {final_acc:.2%}")
    print(f"   Saved to: {model_path}")

    keras.backend.clear_session()
    
    return {
        'produce': produce_name,
        'accuracy': final_acc,
        'classes': class_names
    }

# Train all models
print("\n Training freshness classifiers for all produce types...")
results = []

for produce in PRODUCE_TYPES:
    result = train_freshness_model(produce)
    if result:
        results.append(result)

# Summary
print("\n ALL STAGE 2 MODELS TRAINED!\n")
for r in results:
    print(f"{r['produce'].title():12} | Accuracy: {r['accuracy']:.2%} | Classes: {len(r['classes'])}")
