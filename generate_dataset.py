import os
import random
import glob
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def generate_font_dataset(font_dir):
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    base_output_path = script_dir.parent
    
    extensions = ("*.ttf", "*.otf", "*.TTF", "*.OTF")
    fonts = []
    for ext in extensions:
        fonts.extend(glob.glob(os.path.join(font_dir, ext)))
    
    if not fonts:
        print(f"Check font path: {font_dir}")
        return

    for digit in range(10):
        digit_folder = base_output_path / str(digit)
        digit_folder.mkdir(parents=True, exist_ok=True)
        
        for font_path in fonts:
            font_name = Path(font_path).stem
            
            # 1. Random width and height
            width = random.randint(64, 148)
            height = random.randint(64, 148)
            
            # 2. Create the rectangular canvas
            img = Image.new('L', (width, height), color=0)
            
            try:
                font_scale = min(width, height)
                font_size = int(font_scale * 0.7)
                font = ImageFont.truetype(font_path, font_size)
            except:
                continue

            # 3. Create a temporary square layer for the digit to rotate it safely
            temp_size = max(width, height) * 2 
            temp_img = Image.new('L', (temp_size, temp_size), color=0)
            temp_draw = ImageDraw.Draw(temp_img)
            
            # Center digit on temp square
            bbox = temp_draw.textbbox((0, 0), str(digit), font=font)
            w_text, h_text = bbox[2] - bbox[0], bbox[3] - bbox[1]
            txt_pos = ((temp_size - w_text) // 2, (temp_size - h_text) // 2 - bbox[1])
            temp_draw.text(txt_pos, str(digit), fill=255, font=font)
            
            # 4. Rotate the square temp image
            temp_img = temp_img.rotate(random.uniform(-15, 15), resample=Image.BICUBIC)
            
            # 5. Crop the center of the rotated square and paste onto the random rectangle
            left = (temp_size - width) // 2
            top = (temp_size - height) // 2
            img = temp_img.crop((left, top, left + width, top + height))

            # 6. Add low-power white noise
            img_array = np.array(img).astype(np.float32)
            noise = np.random.normal(0, random.uniform(3, 8), img_array.shape)
            noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            
            final_img = Image.fromarray(noisy_img)
            final_img.save(digit_folder / f"{font_name}.png")

generate_font_dataset("/mnt/c/Windows/Fonts")
