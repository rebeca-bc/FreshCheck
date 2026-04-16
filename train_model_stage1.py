# train_stage1.py
"""
STAGE 1: Produce Type Classifier
Input: Any produce image
Output: strawberry | spinach | tomato | banana | avocado
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt

print("=" * 60)
print("STAGE 1: PRODUCE TYPE CLASSIFIER")
print("=" * 60)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = "data_stages/stage1_classifier" 

# Load data
print("\n Loading produce images...")
train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"Training to identify {len(class_names)} produce types:")
for i, name in enumerate(class_names):
    print(f"   {i+1}. {name.title()}")

# Optimize data pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Build model
print("\nBuilding MobileNetV2 model...")
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base model

# Custom layers on top
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = keras.Model(inputs, outputs)

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n Model Summary:")
model.summary()

# Train
print("\n Training Stage 1 model...")
print("=" * 60)

# for early stopping if this doesbt improve
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=3, # If validation loss doesn't improve for 3 epochs, stop!
    restore_best_weights=True # Rewind to the best epoch
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    verbose=1
)

# Save model
print("\n Saving Stage 1 model...")
import os
os.makedirs('models', exist_ok=True)
model.save('models/stage1_classifier.keras')

# Save class names
with open('models/stage1_classes.txt', 'w') as f:
    f.write('\n'.join(class_names))

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Stage 1: Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Stage 1: Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('stage1_training_history.png')
print(" Training plots saved to stage1_training_history.png")

# Final results
print("\n" + "=" * 60)
print(" STAGE 1 TRAINING COMPLETE!\n")
print(f"Final Training Accuracy:   {history.history['accuracy'][-1]:.2%}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.2%}")
print(f"\nModel saved to: models/stage1_produce_classifier.keras")
print("=" * 60)