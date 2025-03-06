from models.train import train_model  # Import the train_model function from train.py
from models.preProcess import get_data_generators  # Import the get_data_generators function from preprocess.py
import tensorflow as tf

# Define paths
train_dir = "Dataset/task-1/data/train"
validation_dir = "Dataset/task-1/data/validation"

# Get the data generators for training and validation
train_generator, validation_generator = get_data_generators(train_dir, validation_dir)

# -----------------------------------------------------------
# Train the model
# -----------------------------------------------------------
# Comment out all other code that is not related to training
train_model(train_generator, validation_generator)  # Train the model using the generators

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

# -----------------------------------------------------------
# Option 2: Make predictions on new images (Inference)
# -----------------------------------------------------------
# def predict_image(img_path):
#     # Load the trained model
#     model = tf.keras.models.load_model('models/whale_model.h5')

#     # Load and preprocess image
#     img = image.load_img(img_path, target_size=(150, 150))
#     img_array = image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     # Predict
#     prediction = model.predict(img_array)
#     if prediction[0] > 0.5:
#         return "Whale detected!"
#     else:
#         return "No whale detected."

# Uncomment the line below to run inference (replace 'path/to/test_image.jpg' with your image path)
# result = predict_image('path/to/test_image.jpg')
# print(result)

# -----------------------------------------------------------
# Gradio Interface
# -----------------------------------------------------------
# Commented out for now since we are focusing on training
# iface = gr.Interface(
#     fn=predict_whale,  # The function to handle the prediction
#     inputs=gr.Image(type="pil"),  # Updated for Gradio v3.x
#     outputs="text",  # Output will be a text response
#     live=True  # Optional: Update result live while processing
# )
# iface.launch()
