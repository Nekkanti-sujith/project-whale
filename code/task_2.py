import os
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from PIL import Image
import matplotlib.pyplot as plt

# Directories
XVIEW3_IMAGES_DIR = "/Users/sujith/Desktop/project-whale/project-whale/Dataset/task-2/extracted/"
XVIEW3_LABELS_CSV = "/Users/sujith/Desktop/project-whale/project-whale/Dataset/task-2/xview3_labels/train.csv"

# List of specific IDs to process
specific_ids = ["05bc615a9b0e1159t", "72dba3e82f782f67t", "590dd08f71056cacv", "2899cfb18883251bt", "b1844cde847a3942v",
                "cbe4ad26fe73f118t","e98ca5aba8849b06t"]  # Add the specific IDs you want to process
# List of specific IDs to process
specific_ids = ["05bc615a9b0e1159t", "72dba3e82f782f67t", "590dd08f71056cacv", "2899cfb18883251bt", "b1844cde847a3942v",
                "cbe4ad26fe73f118t","e98ca5aba8849b06t"]  # Add the specific IDs you want to process

def convert_csv_to_yolo(csv_path, images_dir, yolo_labels_dir, specific_ids):
    os.makedirs(yolo_labels_dir, exist_ok=True)

def compare_gps_and_detect_ships(csv_path, images_dir, specific_ids):
    # Read CSV file
    annotations = pd.read_csv(csv_path)

    # Filter annotations based on specific_ids
    filtered_annotations = annotations[annotations["scene_id"].isin(specific_ids)]

    # Group by scene_id
    grouped = filtered_annotations.groupby("scene_id")
    grouped = filtered_annotations.groupby("scene_id")

    #to run all the data in csv file
    #grouped= annotations.groupby("scene_id")

    # Initialize counters for metrics
    total_annotations = len(filtered_annotations)
    successful_conversions = 0
    failed_conversions = 0
    missing_scene_dirs = 0
    missing_images = 0

    for scene_id, group in grouped:
        # Locate the directory for the scene_id
        scene_dir = os.path.join(images_dir, scene_id)
        if not os.path.isdir(scene_dir):
            print(f"Warning: Directory for scene_id {scene_id} not found in {images_dir}.")
            continue

        # Look for the specific image file (e.g., '.jpg')
        image_path = os.path.join(scene_dir, "image.jpg")  # Adjust file name if needed
        if not os.path.isfile(image_path):
            print(f"Warning: Image 'image.jpg' for scene_id {scene_id} not found in {scene_dir}.")
            continue

        # Open the image to read it
        img = Image.open(image_path)
        plt.imshow(img)
        plt.title(f"Scene: {scene_id}")
        
        # Loop through the coordinates in the group
        for _, row in group.iterrows():
            lat, lon = row["detect_lat"], row["detect_lon"]
            print(f"Checking coordinates: Lat {lat}, Lon {lon} for scene_id {scene_id}")

            try:
                # Georeference lat/lon to pixel coordinates
                with rasterio.open(image_path) as src:
                    transform = src.transform
                    col, row = rowcol(transform, lon, lat)
                
                # Set a small threshold for the bounding box size
                box_size = 10  # Example size for bounding box (you can adjust)
                x_min = max(0, col - box_size)
                y_min = max(0, row - box_size)
                x_max = min(src.width, col + box_size)
                y_max = min(src.height, row + box_size)

                # If there's a ship, assume a simple logic: check if a ship exists within the box (e.g., using a bounding box)
                if row["is_ship_present"] == 1:  # Assuming 'is_ship_present' column exists in your CSV
                    print(f"Ship detected at coordinates Lat: {lat}, Lon: {lon} in {scene_id}.")
                    
                    # Draw a rectangle on the image to indicate ship's location
                    plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor="r", facecolor="none"))
                    plt.text(x_min, y_min, "Ship", fontsize=12, color='red')

            except Exception as e:
                print(f"Error processing coordinates for {row} -> {e}")
                
        # Show the image with ship annotations
        plt.show()

# Call the function for specific IDs
compare_gps_and_detect_ships(XVIEW3_LABELS_CSV, XVIEW3_IMAGES_DIR, specific_ids)
