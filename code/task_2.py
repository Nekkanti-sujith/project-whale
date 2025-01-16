import os
import json
import rasterio
from rasterio.plot import reshape_as_image
from ultralytics import YOLO
from shapely.geometry import box

# Paths to dataset and output directories
XVIEW3_IMAGES_DIR = "extracted/"      # Directory containing TIFF images
XVIEW3_LABELS_DIR = "xview3_labels/"  # Directory containing GeoJSON annotations
YOLO_LABELS_DIR = "yolo_labels/"      # Directory for YOLO-format labels
OUTPUT_DIR = "output/"                # Directory for saving outputs

# Step 1: Convert GeoJSON annotations to YOLO format
def convert_geojson_to_yolo(geojson_path, image_width, image_height, output_path):
    with open(geojson_path) as f:
        geojson_data = json.load(f)
    features = geojson_data['features']

    with open(output_path, 'w') as out_file:
        for feature in features:
            geometry = feature['geometry']
            properties = feature['properties']

            if geometry['type'] == 'Polygon':
                coords = geometry['coordinates'][0]
                x_min = min([c[0] for c in coords])
                y_min = min([c[1] for c in coords])
                x_max = max([c[0] for c in coords])
                y_max = max([c[1] for c in coords])

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
                obj_class = properties.get('class', 0)

                out_file.write(f"{obj_class} {x_center} {y_center} {width} {height}\n")

# Convert all GeoJSON files to YOLO format
def prepare_yolo_labels():
    os.makedirs(YOLO_LABELS_DIR, exist_ok=True)
    for filename in os.listdir(XVIEW3_LABELS_DIR):
        if filename.endswith('.geojson'):
            geojson_path = os.path.join(XVIEW3_LABELS_DIR, filename)
            tiff_filename = filename.replace('.geojson', '.tif')
            tiff_path = None

            # Search for TIFF in subdirectories
            for root, _, files in os.walk(XVIEW3_IMAGES_DIR):
                if tiff_filename in files:
                    tiff_path = os.path.join(root, tiff_filename)
                    break

            if tiff_path:
                with rasterio.open(tiff_path) as src:
                    height, width = src.height, src.width
                output_path = os.path.join(YOLO_LABELS_DIR, filename.replace('.geojson', '.txt'))
                convert_geojson_to_yolo(geojson_path, width, height, output_path)
            else:
                print(f"Error: Could not find TIFF image {tiff_filename} for annotation {filename}.")

# Step 2: Train the YOLOv8 Model
def train_yolo():
    model = YOLO("yolov8n.pt")  # Load a pre-trained YOLO model
    model.train(
        data="xview3_data.yaml",  # Define this YAML with train/val paths
        epochs=50,
        imgsz=640,
        batch=16,
        project=OUTPUT_DIR,
        name="xview3_training",
    )

# Step 3: Perform Inference with the Trained Model
def detect_ships(image_path, model_path):
    model = YOLO(model_path)  # Load trained YOLO model
    results = model(image_path)

    # Annotate the results
    annotated_image = results[0].plot()
    output_image_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
    cv2.imwrite(output_image_path, annotated_image)
    print(f"Saved annotated image to {output_image_path}")

    # Print detected objects
    for result in results.xyxy[0]:
        x_min, y_min, x_max, y_max, confidence, cls = result
        print(f"Detected class {cls} with confidence {confidence:.2f} at [{x_min}, {y_min}, {x_max}, {y_max}]")

# Step 4: Main Workflow
def main():
    # Step 4.1: Prepare YOLO labels
    print("Converting GeoJSON to YOLO format...")
    prepare_yolo_labels()

    # Step 4.2: Train YOLO model
    print("Training YOLO model on xView3 data...")
    train_yolo()

    # Step 4.3: Detect ships in a sample image
    sample_image = "sample_image.tif"  # Replace with a test image path
    trained_model_path = os.path.join(OUTPUT_DIR, "xview3_training", "weights", "best.pt")
    print("Detecting ships in sample image...")
    detect_ships(sample_image, trained_model_path)

if __name__ == "__main__":
    main()
