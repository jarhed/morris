import torch
from transformers import pipeline
from PIL import Image
import os

# Initialize the depth estimation pipeline
depth_estimator = pipeline(task="depth-estimation", model="Intel/dpt-large")

# Load the image
image_path = "image_expanded.png"
image = Image.open(image_path)

# Estimate depth
result = depth_estimator(image)
depth_map = result["depth"]

# Save the depth map
depth_map.save("depth_huggingface.png")
print("Depth map saved to depth_huggingface.png")
