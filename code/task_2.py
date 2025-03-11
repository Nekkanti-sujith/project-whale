import gradio as gr
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw
import requests
import numpy as np

# Load model and processor
model_name = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)

# Inference function
def inference(image):
    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")
    # Perform model inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process the output and extract bounding boxes
    target_sizes = torch.tensor([image.size[::-1]])  # Convert to (height, width)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

    # Draw bounding boxes on the image
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    detected = False  # Flag to check if any object is detected
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}", fill="red")
        detected = True

    # Return image with boxes and detection message
    if detected:
        return image_with_boxes, "Ship(s) detected!"
    else:
        return image_with_boxes, "No ship detected."

# Gradio interface setup
# iface = gr.Interface(
#     fn=inference,
#     inputs=gr.Image(type="pil"),
#     outputs=[gr.Image(type="pil"), gr.Textbox()],
#     title="Object Detection with DETR",
#     description="Upload an image to detect ships using the DETR model."
# )

# # Launch Gradio interface with a public link
# iface.launch(share=False)
