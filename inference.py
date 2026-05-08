"""
inference.py

Usage:
    python inference.py <image_path> --model <path_to_model.npz>
"""

import sys
import argparse
import numpy as np
import pre_process as pp
import calculation_functions as cf

def load_model(model_path):
    model = np.load(model_path)
    filters, conv_bias, weights, fc_bias = model["filter_bank"], model["conv_bias"], model["fc_weights"], model["fc_bias"]
    return filters, conv_bias, weights, fc_bias

def image_preprocess(image_path):
    img = pp.load_image(image_path)
    resized = pp.resize_image(img)
    image_array = pp.to_numpy(resized)
    return image_array

def infer(image_array, filter_bank, conv_bias, fc_weights, fc_bias):
    probs, _ = cf.forward_pass(image_array, filter_bank, conv_bias, fc_weights, fc_bias)
    return probs

def print_results(probs):
    for digit, prob in enumerate(probs):
        # Convert the prob to a percentage format
        print(f"  {digit}: {prob * 100:.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--model", default="models/final_model.npz")
    args = parser.parse_args()

    filter_bank, conv_bias, fc_weights, fc_bias = load_model(args.model)
    image_array = image_preprocess(args.image_path)
    probs = infer(image_array, filter_bank, conv_bias, fc_weights, fc_bias)
    print_results(probs)

if __name__ == "__main__":
    main()