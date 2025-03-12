# import torch
# import json
# import os
# from PIL import Image
# from transformers import DetrForObjectDetection, DetrImageProcessor
# from datasets import Dataset
# from sklearn.model_selection import train_test_split

# # Define paths
# ANNOTATIONS_JSON = "output/task-2/annotations_coco.json"  # Update this path
# IMAGE_DIR = "Dataset/task-2/ships"  # Update this path

# # Load DETR model and processor
# processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
# model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# # Load COCO annotations
# def load_annotations_coco(json_file):
#     with open(json_file, 'r') as f:
#         return json.load(f)

# coco_data = load_annotations_coco(ANNOTATIONS_JSON)

# # Convert COCO format to Hugging Face dataset format
# def prepare_dataset():
#     images = []
#     annotations = []
    
#     for img in coco_data["images"]:
#         image_id = img["id"]
#         image_path = os.path.join(IMAGE_DIR, img["file_name"])
        
#         if not os.path.exists(image_path):  # Skip missing images
#             continue
        
#         image_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]
        
#         images.append({"image_path": image_path, "annotations": image_annotations})

#     return Dataset.from_list(images)

# dataset = prepare_dataset()

# # Function to process images and annotations
# def process_images_and_annotations(example):
#     image = Image.open(example["image_path"]).convert("RGB")

#     # Transform the image for DETR
#     inputs = processor(images=image, return_tensors="pt")

#     # Process annotations
#     annotations = example["annotations"]
#     targets = []
    
#     for ann in annotations:
#         box = ann["bbox"]  # Format: [x_min, y_min, width, height]
#         label = ann["category_id"]
#         targets.append({"bbox": box, "label": label})

#     return {
#         "pixel_values": inputs["pixel_values"].squeeze(0),  # Remove batch dimension
#         "labels": targets
#     }

# # Apply transformations to dataset
# train_size = 0.8  # Set the train size ratio
# dataset_list = dataset.to_list()  # Convert dataset to a list
# train_list, eval_list = train_test_split(dataset_list, train_size=train_size, random_state=42)

# train_dataset = Dataset.from_list(train_list)
# eval_dataset = Dataset.from_list(eval_list)

# train_dataset = train_dataset.map(process_images_and_annotations, remove_columns=["image_path", "annotations"])
# eval_dataset = eval_dataset.map(process_images_and_annotations, remove_columns=["image_path", "annotations"])

# # Save the processed datasets to disk (optional)
# train_dataset.save_to_disk("train_dataset")  # Save the train dataset
# eval_dataset.save_to_disk("eval_dataset")  # Save the eval dataset

# # Training setup
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# # Example batch for debugging
# batch = train_dataset[0]
# print("Sample batch:", batch)

# # After training is done, save the model
# model.save_pretrained("output/task-2")  # Replace with your desired path
# processor.save_pretrained("output/task-2")  # Save processor (tokenizer, etc.)

# # Training loop placeholder (replace with full training logic)
# print("Dataset processed successfully! Ready for training.")