import os
import pandas as pd
import rasterio
from rasterio.transform import rowcol

# Directories
XVIEW3_IMAGES_DIR = "/Users/sujith/Desktop/project-whale/project-whale/Dataset/task-2/extracted/"
XVIEW3_LABELS_CSV = "/Users/sujith/Desktop/project-whale/project-whale/Dataset/task-2/xview3_labels/train.csv"
YOLO_LABELS_DIR = "/Users/sujith/Desktop/project-whale/project-whale/Dataset/task-2/yolo_labels"

# List of specific IDs to process
specific_ids = ["05bc615a9b0e1159t", "72dba3e82f782f67t", "590dd08f71056cacv", "2899cfb18883251bt", "b1844cde847a3942v",
                "cbe4ad26fe73f118t","e98ca5aba8849b06t"]  # Add the specific IDs you want to process

def convert_csv_to_yolo(csv_path, images_dir, yolo_labels_dir, specific_ids):
    os.makedirs(yolo_labels_dir, exist_ok=True)

    # Read CSV file
    annotations = pd.read_csv(csv_path)

# conditions for only filter id's, bceasue of small dataset

    # Filter annotations based on specific_ids
    filtered_annotations = annotations[annotations["scene_id"].isin(specific_ids)]

    # Group by scene_id
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
            missing_scene_dirs += len(group)  # Count all rows in this group as failed
            continue

        # Look for the specific TIFF file (e.g., 'VV_dB.tif')
        image_path = os.path.join(scene_dir, "VV_dB.tif")  # Adjust file name if needed
        if not os.path.isfile(image_path):
            print(f"Warning: Image 'VV_dB.tif' for scene_id {scene_id} not found in {scene_dir}.")
            missing_images += len(group)  # Count all rows in this group as failed
            continue

        # Open the image to read geospatial info
        with rasterio.open(image_path) as src:
            transform = src.transform
            image_width, image_height = src.width, src.height

        # Create YOLO annotation file
        yolo_file = os.path.join(yolo_labels_dir, f"{scene_id}.txt")
        with open(yolo_file, "w") as f:
            for _, row in group.iterrows():
                try:
                    # Georeference lat/lon to pixel coordinates
                    lat, lon = row["detect_lat"], row["detect_lon"]
                    row_col = rowcol(transform, lon, lat)
                    col, row = row_col

                    # Define a placeholder bounding box
                    box_size = 5  # Example size for bounding box
                    x_min = max(0, col - box_size)
                    y_min = max(0, row - box_size)
                    x_max = min(image_width, col + box_size)
                    y_max = min(image_height, row + box_size)

                    # Normalize coordinates
                    x_min_norm = x_min / image_width
                    y_min_norm = y_min / image_height
                    x_max_norm = x_max / image_width
                    y_max_norm = y_max / image_height

                    # YOLO format: class x_center y_center width height
                    x_center = (x_min_norm + x_max_norm) / 2
                    y_center = (y_min_norm + y_max_norm) / 2
                    width = x_max_norm - x_min_norm
                    height = y_max_norm - y_min_norm

                    # Default class = 0 (can be modified)
                    f.write(f"0 {x_center} {y_center} {width} {height}\n")
                    successful_conversions += 1
                except Exception as e:
                    print(f"Error processing lat/lon for row: {row} -> {e}")
                    failed_conversions += 1

        print(f"Converted annotations for {scene_id} to YOLO format.")

    # Calculate and print accuracy metrics
    total_failed = missing_scene_dirs + missing_images + failed_conversions
    accuracy = (successful_conversions / total_annotations) * 100 if total_annotations > 0 else 0
    print("\n--- Conversion Summary ---")
    print(f"Total Annotations: {total_annotations}")
    print(f"Successful Conversions: {successful_conversions}")
    print(f"Failed Conversions: {failed_conversions}")
    print(f"Missing Scene Directories: {missing_scene_dirs}")
    print(f"Missing Images: {missing_images}")
    print(f"Accuracy: {accuracy:.2f}%")

# Call the function for specific IDs
convert_csv_to_yolo(XVIEW3_LABELS_CSV, XVIEW3_IMAGES_DIR, YOLO_LABELS_DIR, specific_ids)
