import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import csv

MODEL_PATH = "models/whale_model.h5"  # Path to trained model
NEW_IMAGES_DIR = "new_images"         # Folder for unknown images
OUTPUT_FILE = "whale_gps_data.csv"    # CSV file for storing GPS data

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

def load_and_preprocess_image(image_path):
    """Load an image, preprocess for model prediction."""
    img = image.load_img(image_path, target_size=(150, 150))  # Resize to model input size
    img_array = image.img_to_array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

def get_exif_data(image_path):
    """ Extract EXIF metadata from an image. """
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            return None
        return {TAGS.get(tag, tag): value for tag, value in exif_data.items()}
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

def get_gps_coordinates(exif_data):
    """ Extract GPS coordinates from EXIF metadata. """
    if "GPSInfo" not in exif_data:
        return None
    
    gps_info = exif_data["GPSInfo"]
    gps_data = {GPSTAGS.get(tag, tag): value for tag, value in gps_info.items()}

    if "GPSLatitude" in gps_data and "GPSLongitude" in gps_data:
        lat = convert_to_degrees(gps_data["GPSLatitude"])
        lon = convert_to_degrees(gps_data["GPSLongitude"])

        if gps_data.get("GPSLatitudeRef") == "S":
            lat = -lat
        if gps_data.get("GPSLongitudeRef") == "W":
            lon = -lon

        return lat, lon
    return None

def convert_to_degrees(value):
    """ Convert DMS (degrees, minutes, seconds) to decimal degrees. """
    d, m, s = value
    return d + (m / 60.0) + (s / 3600.0)

def is_in_ocean(lat, lon):
    """ Check if the GPS coordinates are in the Pacific or Atlantic Ocean. """
    if -180 <= lon <= -70 and -60 <= lat <= 60:
        return "Pacific Ocean"
    if -70 <= lon <= 20 and -60 <= lat <= 60:
        return "Atlantic Ocean"
    return None

def classify_and_extract_gps(directory, output_file):
    """Predict images, extract GPS for whales, and store data."""
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["File Name", "Latitude", "Longitude", "Ocean Name"])  # Writing header

        for filename in os.listdir(directory):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(directory, filename)
                
                # Run prediction
                img_array = load_and_preprocess_image(image_path)
                prediction = model.predict(img_array)[0][0]  # Binary classification
                
                if prediction > 0.5:  # Whale detected
                    print(f"Whale detected in {filename}")

                    # Extract GPS metadata
                    exif_data = get_exif_data(image_path)
                    if exif_data:
                        gps_coords = get_gps_coordinates(exif_data)
                        if gps_coords:
                            ocean = is_in_ocean(*gps_coords)
                            if ocean:
                                writer.writerow([filename, gps_coords[0], gps_coords[1], ocean])
                                print(f"Saved: {filename} - {gps_coords[0]}, {gps_coords[1]} ({ocean})")

# Run inference
if __name__ == "__main__":
    classify_and_extract_gps(NEW_IMAGES_DIR, OUTPUT_FILE)
    print("Inference and GPS extraction completed.")
