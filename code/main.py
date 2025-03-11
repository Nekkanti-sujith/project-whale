import gradio as gr

from task_2 import inference  # Ship Detection
from task_1_final import predict_image  # Whale Detection

import os
port = os.getenv('PORT', 8080)

# Function to reveal image uploader and hide button
def show_uploader():
    return gr.update(visible=False), gr.update(visible=True)

# Function to process image through both models
def process_image(image):
    _, whale_output = predict_image(image)  # Whale detection
    _, ship_output = inference(image)  # Ship detection

    # Combine outputs into one formatted string
    total_output = f"**Whale Detection:** {whale_output}\n**Ship Detection:** {ship_output}"
    
    # Show location input and submit button only if a whale is detected
    show_location = whale_output.lower().strip() != "no whale detected."
    return total_output, gr.update(visible=show_location), gr.update(visible=show_location), gr.update(visible=False)

# Function to handle location submission and save to file
def submit_location(location):
    with open("whale_reports.txt", "a") as file:
        file.write(f"Reported Whale Location: {location}\n")
    return "Thank you for choosing to save whales!", gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

# Gradio Interface
with gr.Blocks() as demo:
    with gr.Column():
        button = gr.Button("Press to Save Whales")
        
        # Display sample images as hyperlinks
        sample_images = """
        Here are some sample images:
        - [Sample Image 1](https://sujith999awsbucket.s3.us-east-2.amazonaws.com/Test+detection+_+w1.png)
        - [Sample Image 2](https://sujith999awsbucket.s3.us-east-2.amazonaws.com/Test+detection+_+w249.png)
        - [Sample Image 3](https://sujith999awsbucket.s3.us-east-2.amazonaws.com/Test+detection+_+w911.png)
        """
        gr.Markdown(sample_images)

    with gr.Row(visible=False) as upload_section:
        image_input = gr.Image(type="pil")  # Image uploader
    
    output_text = gr.Markdown()
    location_input = gr.Textbox(placeholder="Enter location", visible=False)
    submit_button = gr.Button("Submit", visible=False)
    thank_you_text = gr.Markdown(visible=False)

    # Button Click: Hide button, Show uploader
    button.click(fn=show_uploader, inputs=None, outputs=[button, upload_section])

    # Image Upload: Run both models and show result
    image_input.change(fn=process_image, inputs=image_input, outputs=[output_text, location_input, submit_button, thank_you_text])

    # Submit location, save to file, show thank you message, hide input fields
    submit_button.click(fn=submit_location, inputs=location_input, outputs=[thank_you_text, location_input, submit_button, thank_you_text])
    
# Launch
demo.launch(server_name="0.0.0.0",server_port=port)

