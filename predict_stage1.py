import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
import os

# Suppress annoying TensorFlow warnings in the terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

print("\n STAGE 1: testing the inference engine\n")

# Load saved model and the class names
print("Loading model...")
model = keras.models.load_model('models/stage1_classifier.keras')

with open('models/stage1_classes.txt', 'r') as f:
    class_names = f.read().splitlines()

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f" Error: Could not find image at '{image_path}'")
        return

    print(f"\n Analyzing: {image_path}")

    #  Load the image and force it to be 224x224 
    img = keras.utils.load_img(image_path, target_size=(224, 224))

    # Convert the image into a 3D grid of numbers (Height, Width, Colors)
    img_array = keras.utils.img_to_array(img)

    # The "Batch" Trick, because its only 1 image 
    # We use np.expand_dims to put our 1 image into an invisible batch of 1.
    img_batch = np.expand_dims(img_array, axis=0)

    # Make the prediction! with probabilities
    predictions = model.predict(img_batch, verbose=0)
    confidence_scores = predictions[0] # Grab the results for our single image

    # Find the highest probability
    predicted_index = np.argmax(confidence_scores)
    predicted_class = class_names[predicted_index]
    confidence = confidence_scores[predicted_index] * 100

    # Print the final verdict
    print("\n RESULTS\n")
    print(f"Prediction : {predicted_class.upper()}")
    print(f"Confidence : {confidence:.2f}%\n")

    # Show the breakdown of what else it thought it might be
    print("Breakdown:")
    for i, score in enumerate(confidence_scores):
        print(f" - {class_names[i].title().ljust(15)}: {score*100:>6.2f}%")

# Run the test! 
if __name__ == "__main__":
    test_file = "test_img.png" 
    predict_image(test_file)