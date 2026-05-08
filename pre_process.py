import numpy as np
from PIL import Image
import sys

image_path = sys.argv[1] if len(sys.argv) > 1 else None


def load_image(image_path):
    return Image.open(image_path).convert('1')


def resize_image_preserve_aspect_ratio(image):
    if isinstance(image, str):
        with Image.open(image) as img:
            width, height = img.size
    else:
        width, height = image.size

    if width > height:
        new_width = 28
        new_height = int(height * (new_width / width))
    else:
        new_height = 28
        new_width = int(width * (new_height / height))

    img_resized = image.resize((new_width, new_height), Image.LANCZOS)

    left_pad = (28 - new_width) // 2
    top_pad = (28 - new_height) // 2

    img_final = Image.new('RGB', (28, 28), color='black')
    img_final.paste(img_resized, (left_pad, top_pad))

    return img_final.convert('L')


def to_numpy(image):
    arr = np.array(image, dtype=np.float32)
    arr /= 255.0
    return arr


if __name__ == "__main__":
    if image_path is None:
        sys.exit("Usage: python pre_process.py <image_path>")

    img = load_image(image_path)
    resized = resize_image_preserve_aspect_ratio(img)
    image_array = to_numpy(resized)

    Image.fromarray(image_array).save("image_output.png")
    print(f"Image Array Shape: {image_array.shape}")
