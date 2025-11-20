import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import numpy as np

model_name = "depth-anything/Depth-Anything-V2-Large-hf"
image_processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForDepthEstimation.from_pretrained(model_name)

image_path = "image_expanded.png"
image = Image.open(image_path)

inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

depth_array = prediction.squeeze().cpu().numpy()
depth_array = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
depth_map = Image.fromarray((depth_array * 255).astype(np.uint8))

depth_map.save("depth_huggingface.png")
print("Depth map saved to depth_huggingface.png")
