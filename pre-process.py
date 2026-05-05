# ========================================================================================
"""
Overview: Preprocessing 
Preprocssing Images of different font numbers for CNN analysis

Removes noise, converts image of digit to pure black/white, resizes, and converts the image to a NumPy array
"""
# ========================================================================================
"""
Table of Contents:
1. Arguments and General Variables
2. Converting to Black/White
3. Re-size image to 28x28
4. Convert Image to Grayscale Numpy Array Values
"""
# ========================================================================================
# Step 1: Arguments and imports
# ========================================================================================
import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image
import sys
image_path = sys.argv[1]









# ========================================================================================
# Step 2: Converting to Black/White (Or grayscale
# ========================================================================================
img = Image.open(image_path)
# I'm using '1' for pure black/white, but you can use 'L' if you want, which converts it to grayscale
img_grayscale = img.convert('1')







# ========================================================================================
# Step 3: Resize Image without distortion
# ========================================================================================


def resize_image_preserve_aspect_ratio(image):
    if isinstance(image, str):
        with Image.open(image) as img:
            width, height = img.size
    else:
        width, height = image.size
    
    # Calculate new size (while maintaining image ratio)
    if width > height:
        new_width = 28
        new_height = int(height * (new_width / width))
    else:
        new_height = 28
        new_width = int(width * (new_height / height))
    
    # Resize image
    img_resized = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Calculate padding to center the image
    left_pad = (28 - new_width) // 2
    top_pad = (28 - new_height) // 2
    right_pad = 28 - new_width - left_pad
    bottom_pad = 28 - new_height - top_pad
    
    # Create a new image with white background and paste the resized image onto it
    img_final = Image.new('RGB', (28, 28), color='black')
    img_final.paste(img_resized, (left_pad, top_pad))
    
    # Convert back to grayscale
    img_final = img_final.convert('L')
    
    return img_final

resized_image = resize_image_preserve_aspect_ratio(img_grayscale)

# ========================================================================================
# Step 4: Make a Numpy Array of the 28x28 pixel processed image
# ========================================================================================
image_array = np.array(resized_image)
Image.fromarray(image_array).save("image_output.png") # Debug
print(f"Image Array Shape: {image_array.shape}")  # Debug