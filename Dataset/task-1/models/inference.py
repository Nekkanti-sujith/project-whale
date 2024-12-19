import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model('models/whale_model.h5')

def predict_image(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        return "Whale detected!"
    else:
        return "No whale detected."

# Test with a new image
img_path = 'path/to/test_image.jpg'
result = predict_image(img_path)
print(result)