# pre_process.py
# =============================================================================
"""
Overview: Preprocessing Images of different font numbers for CNN analysis

Removes noise, converts image of digit to pure black/white, resizes,
and converts the image to a NumPy array.
"""
# =============================================================================
"""
Table of Contents:
1. Arguments and General Variables
2. Converting to Black/White
3. Re-size image to 28x28
4. Convert Image to Grayscale Numpy Array Values
"""
# =============================================================================
# Step 1: Arguments and imports
# =============================================================================
import numpy as np
from PIL import Image
import sys

# ---- keep the original variable (used when the script is run directly) ----
image_path = sys.argv[1] if len(sys.argv) > 1 else None   # <-- unchanged name

# =============================================================================
# Step 2: Converting to Black/White (Or grayscale) and then make them just 0's or 1's
# =============================================================================
def _load_image(image_path):
    """Internal helper used both by the script block and by imports."""
    img = Image.open(image_path)
    # '1' gives pure black/white (already what the original script used)
    return img.convert('1')

# =============================================================================
# Step 3: Resize Image without distortion
# =============================================================================
def resize_image_preserve_aspect_ratio(image):
    """Resize a PIL image to 28×28 while preserving aspect ratio and padding."""
    if isinstance(image, str):
        with Image.open(image) as img:
            width, height = img.size
    else:
        width, height = image.size

    # ---- keep the original logic exactly ----
    if width > height:
        new_width = 28
        new_height = int(height * (new_width / width))
    else:
        new_height = 28
        new_width = int(width * (new_height / height))

    img_resized = image.resize((new_width, new_height), Image.LANCZOS)

    left_pad = (28 - new_width) // 2
    top_pad = (28 - new_height) // 2
    right_pad = 28 - new_width - left_pad
    bottom_pad = 28 - new_height - top_pad

    # original script created a black background (RGB black → later converted)
    img_final = Image.new('RGB', (28, 28), color='black')
    img_final.paste(img_resized, (left_pad, top_pad))

    # convert back to single‑channel grayscale (the original used 'L')
    return img_final.convert('L')

# =============================================================================
# Step 4: Make a Numpy Array of the 28x28 pixel processed image
# =============================================================================
def _to_numpy(image):
    arr = np.array(image, dtype=np.float32)
    arr /= 255.0
    return arr

# =============================================================================
#  Script‑only execution block (kept for backward compatibility)
# =============================================================================
if __name__ == "__main__":
    # When run directly we expect an image path; the original script would
    # have raised IndexError otherwise, so we preserve that behaviour.
    if image_path is None:
        sys.exit("Usage: python pre_process.py <image_path>")

    # Original pipeline (unchanged logic, just wrapped in functions)
    img = _load_image(image_path)                       # step 2
    resized = resize_image_preserve_aspect_ratio(img)  # step 3
    image_array = _to_numpy(resized)                    # step 4

    # Debug output – identical to the original script
    Image.fromarray(image_array).save("image_output.png")
    print(f"Image Array Shape: {image_array.shape}")
