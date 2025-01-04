import os
from dotenv import load_dotenv
import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageEnhance

# Load environment variables
load_dotenv()

# Retrieve environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "runwayml/stable-diffusion-v1-5")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
NUM_STEPS = int(os.getenv("NUM_STEPS", 50))

# Load pre-trained model
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

# Function to generate images from text
def generate_image(prompt):
    try:
        image = pipe(prompt, num_inference_steps=NUM_STEPS).images[0]
        # Post-processing: Enhance image quality
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)  # Increase contrast
        return image
    except Exception as e:
        print(f"Error generating image for prompt '{prompt}': {e}")
        return Image.new("RGB", (512, 512), "white")  # Return a blank image on error

# Function to create a comic strip from multiple images
def create_comic_strip(text_input):
    frames = text_input.split("\n")
    images = [generate_image(frame) for frame in frames]
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths) + len(images) * 10  # Add space between frames
    max_height = max(heights)
    comic_strip = Image.new("RGB", (total_width, max_height), "white")
    
    x_offset = 0
    for img in images:
        comic_strip.paste(img, (x_offset, 0))
        x_offset += img.width + 10  # Add space between frames
    return comic_strip

# Define Gradio interface
def generate_comic(text_input):
    return create_comic_strip(text_input)

# Gradio app
interface = gr.Interface(
    fn=generate_comic,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Enter comic script. Each line corresponds to a frame.",
        label="Comic Script",
    ),
    outputs="image",
    title="Comic Book Generator",
    description="Enter your comic script. Each line represents a frame of the comic strip. More descriptive prompts yield better results!",
)

# Launch app
interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
