import numpy as np
from PIL import Image
import sys

image_path = sys.argv[1] if len(sys.argv) > 1 else None


def load_image(image_path):
    return Image.open(image_path).convert('1')


def resize_image(image):
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

    img_final = Image.new('1', (28, 28), color=0)
    img_final.paste(img_resized, (left_pad, top_pad))

    return img_final


def to_numpy(image):
    arr = np.array(image, dtype=np.float32)
    return arr


def main(image_path_arg=None):
    path = image_path_arg or image_path
    if path is None:
        sys.exit("Usage: python pre_process.py <image_path>")

    img = load_image(path)
    resized = resize_image(img)
    image_array = to_numpy(resized)
    print(f"Image Array Shape: {image_array.shape}")
    return image_array

if __name__ == "__main__":
    main()
