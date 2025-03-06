import gradio as gr
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw
import torch

# Load the pre-trained DETR model
model_name = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)

# General categories of sea-based transportation (ship and boat)
TRANSPORTATION_LABELS = [7, 9]  # [7: ship, 9: boat]

# Function to predict if the uploaded image contains any mode of sea transportation (ship, boat, etc.)
def predict_transport_in_image(img: Image.Image):
    try:
        # Preprocess the image
        inputs = processor(images=img, return_tensors="pt")
        
        # Perform inference to detect objects
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get the predicted boxes and labels
        target_sizes = torch.tensor([img.size[::-1]])  # Convert (width, height) to (height, width)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
        
        # Initialize ImageDraw object to draw bounding boxes
        draw = ImageDraw.Draw(img)
        found_transport = False

        # Check if any sea-based transportation is detected and draw the bounding box
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if label in TRANSPORTATION_LABELS:
                found_transport = True
                box = [round(i, 2) for i in box.tolist()]  # Round the coordinates for better readability
                draw.rectangle(box, outline="red", width=3)  # Draw bounding box (red color)
                # Add label to the box
                draw.text((box[0], box[1]), "Ship/Boat", fill="red")
        
        # If relevant transportation is detected, return message and the image
        if found_transport:
            return img, "The image contains a ship, boat, etc!"  
        else:
            return img, "The image does not contain a ship, boat, etc."
    
    except Exception as e:
        return img, f"Error: {str(e)}"

# Create Gradio interface
interface = gr.Interface(
    fn=predict_transport_in_image,  # Function to run on image input
    inputs=gr.Image(type="pil"),  # Input: Image upload
    outputs=["image", "text"],  # Output: Image with bounding box and text message
    live=True,  # Optional: enable live feedback
)

# Launch the Gradio interface
interface.launch()
