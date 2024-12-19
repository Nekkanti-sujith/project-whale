import tensorflow as tf
from preprocess import get_data_generators

# Load model
model = tf.keras.models.load_model('models/whale_model.h5')

# Define paths
train_dir = "data/train"
validation_dir = "data/validation"

# Get data generators
_, validation_generator = get_data_generators(train_dir, validation_dir)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy:.2f}")
print(f"Validation Loss: {loss:.2f}")