import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import gradio as gr

# -----------------------------------------------------------
# Load the trained model
# -----------------------------------------------------------
model = tf.keras.models.load_model('/Users/sujith/Desktop/project-whale/project-whale/output/task-1/whale_model.h5')

# -----------------------------------------------------------
# Option 2: Make predictions on new images (Inference)
# -----------------------------------------------------------
def predict_whale(img):
    # Load and preprocess image
    img = img.resize((150, 150))  # Resize image to match the model input size
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        return "Whale detected!"
    else:
        return "No whale detected."

# -----------------------------------------------------------
# Gradio Interface
# -----------------------------------------------------------
iface = gr.Interface(
    fn=predict_whale,  # The function to handle the prediction
    inputs=gr.Image(type="pil"),  # Updated for Gradio v3.x
    outputs="text",  # Output will be a text response
    live=True  # Optional: Update result live while processing
)

# Launch the interface
iface.launch()

# -----------------------------------------------------------
# Option 1: Evaluate the model on the validation set
# -----------------------------------------------------------
# def evaluate_model():
#     # Load the trained model
#     model = load_model('output/task-1/whale_model.h5')

#     # Evaluate the model
#     loss, accuracy = model.evaluate(validation_generator)
#     print(f"Validation Accuracy: {accuracy:.2f}")
#     print(f"Validation Loss: {loss:.2f}")

# Uncomment the line below to run evaluation
# evaluate_model()
