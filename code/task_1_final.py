import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing import image

# Load the trained model once to avoid reloading on every function call
model = tf.keras.models.load_model('output/task-1/custom_whale_model.keras')

def predict_image(img):
    # Convert PIL image to numpy array
    img = img.resize((150, 150))  # Resize image to match model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict whale presence
    prediction = model.predict(img_array)
    
    # Create a copy of the image to draw bounding boxes
    image_with_boxes = img.copy()
    draw = ImageDraw.Draw(image_with_boxes)

    # Set detection message based on prediction
    if prediction[0] > 0.5:
        detection_message = "Whale detected!"
        # Draw a bounding box around the detected whale
        draw.rectangle([20, 20, 130, 130], outline="blue", width=3)  # Example box
        draw.text((20, 20), detection_message, fill="blue")
    else:
        detection_message = "No whale detected."

    # Return the image with boxes and detection message
    return image_with_boxes, detection_message

