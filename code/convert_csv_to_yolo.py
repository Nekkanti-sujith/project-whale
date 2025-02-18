import os
import pandas as pd
import rasterio
from rasterio.transform import rowcol

# Directories
XVIEW3_IMAGES_DIR = "/Users/sujith/Desktop/project-whale/project-whale/Dataset/task-2/train_images"  
XVIEW3_LABELS_CSV = "/Users/sujith/Desktop/project-whale/project-whale/Dataset/task-2/xview3_labels/train.csv"  
YOLO_LABELS_DIR = "/Users/sujith/Desktop/project-whale/project-whale/Dataset/task-2/train_labels"  

def convert_csv_to_yolo(csv_path, images_dir, yolo_labels_dir):
    os.makedirs(yolo_labels_dir, exist_ok=True)

    # Read CSV file
    annotations = pd.read_csv(csv_path)

    # Keep only rows where 'is_vessel' is True or False
    annotations = annotations.dropna(subset=["is_vessel"])
    
    # Convert is_vessel to numerical class labels (1 = ship, 0 = no ship)
    annotations["class_id"] = annotations["is_vessel"].astype(int)

    # Group annotations by scene_id
    grouped = annotations.groupby("scene_id")

    total_annotations = len(annotations)
    successful_conversions = 0
    failed_conversions = 0

    for scene_id, group in grouped:
        scene_dir = os.path.join(images_dir, scene_id)
        if not os.path.isdir(scene_dir):
            print(f"Warning: Directory for scene_id {scene_id} not found.")
            continue

        tif_files = [f for f in os.listdir(scene_dir) if f.endswith('.tif')]
        for tif_file in tif_files:
            image_path = os.path.join(scene_dir, tif_file)

            # Open the image to read geospatial info
            with rasterio.open(image_path) as src:
                transform = src.transform
                image_width, image_height = src.width, src.height

            # Create YOLO annotation file
            yolo_file = os.path.join(yolo_labels_dir, f"{scene_id}_{tif_file}.txt")
            with open(yolo_file, "w") as f:
                for _, row in group.iterrows():
                    try:
                        lat, lon = row["detect_lat"], row["detect_lon"]
                        row_col = rowcol(transform, lon, lat)
                        col, row = row_col

                        # Define bounding box size
                        box_size = 5  
                        x_min = max(0, col - box_size)
                        y_min = max(0, row - box_size)
                        x_max = min(image_width, col + box_size)
                        y_max = min(image_height, row + box_size)

                        # Normalize coordinates
                        x_center = ((x_min + x_max) / 2) / image_width
                        y_center = ((y_min + y_max) / 2) / image_height
                        width = (x_max - x_min) / image_width
                        height = (y_max - y_min) / image_height

                        # Get class label (1 for ship, 0 for non-ship)
                        class_id = int(row["class_id"])

                        # Write in YOLO format
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                        successful_conversions += 1
                    except Exception as e:
                        print(f"Error processing row: {row} -> {e}")
                        failed_conversions += 1

            print(f"Converted {scene_id} - {tif_file} to YOLO format.")

    # Summary
    print("\n--- Conversion Summary ---")
    print(f"Total Annotations: {total_annotations}")
    print(f"Successful Conversions: {successful_conversions}")
    print(f"Failed Conversions: {failed_conversions}")

# Run conversion
convert_csv_to_yolo(XVIEW3_LABELS_CSV, XVIEW3_IMAGES_DIR, YOLO_LABELS_DIR)
