import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image  # Import PIL

# Load the trained model globally to avoid reloading it every time
model = tf.keras.models.load_model('/Users/sujith/Desktop/project-whale/project-whale/output/task-1/custom_whale_model.keras')

def predict_image(img):
    # Convert to PIL Image if Gradio provides a NumPy array
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)  # Convert NumPy array to PIL image

    # Resize the image
    img = img.resize((150, 150))  # Resize the image

    # Convert image to NumPy array
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for model input

    # Predict
    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        return "Whale detected!"
    else:
        return "No whale detected."

# Create Gradio interface
iface = gr.Interface(
    fn=predict_image,  # The function to handle the prediction
    inputs=gr.Image(),  # Gradio will handle image conversion
    outputs="text",  # Output will be a text response
)

# Launch the Gradio app
iface.launch()