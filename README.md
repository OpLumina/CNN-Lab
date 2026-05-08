## CNN Built from Scratch Project
## Goals
Coding a CNN using only basic python libraries like NumPy, pandas, and PIL for basic digit classification from different fonts

## 5/4/2026:
This is the start of a personal project I'm coding with a few people to understand CNN's better.

**Done Today:**
* 1-bit Image Color Normalization (black=1, white=0) **(pre-process.py)**
* Resizing to 28x28 while keeping image ratios the same **(pre-process.py)**
* Turning the Processed image into a NumPy Array **(pre-process.py)**
* Dataset Generation **(generate_dataset.py)**

**Decided to change from that dataset to a different idea of making my own with different fonts and adding basic noise**

**To Do:**
* Feature extraction: 3x3 sliding window

## 5/5/2026:
**Done Today:**
* Feature Extraction Function **(calculation_functions.py)** with args for window instead of a set 3x3
* Added a Dataset generation script that pulls from windows fonts and generates a few different datasets for numbers 0-9

**To Do:**
* Max Pooling Function

## 5/6/2026-5/7/2026:
**Done These Days:**
* Feature extraction bug fix on counters
* Filter and weight initialization
* Max pooling fixes

**Feature Map now handles 3D filter matrices:**
Extracted kernel is multiplied by a 3D filter array (filter_depth, height, width), appended to a list of each extraction during convolution (4D array), reshaped to a 5D array (matrix height, matrix width, matrix depth), sums each 2D matrix to convert the 5D array to a 3D array, applies ReLU on each value, then adds bias to produce a 3D feature map of all filtered outputs. Pre-ReLU values are cached for use in backpropagation.

* Pooling handles 3D array output of feature map
* Added full backpropagation with gradient tracking through conv, pool, ReLU, and fully connected layers
* Fixed relu and relu_backward — relu now uses np.maximum instead of Python's max(), and relu_backward now correctly uses cached pre-ReLU values instead of post-ReLU
* Added softmax and cross entropy loss
* Added fully connected (linear) layer with forward and backward pass
* Added forward_pass and backward_pass end-to-end functions in **(calculation_functions.py)**
* Added **train.py** with full training loop:
    * Loads and preprocesses dataset from ./dataset/<digit>/
    * Shuffles data before train/val split to ensure all digits are represented in validation
    * 80/20 train/validation split
    * Mini-batch gradient descent with configurable batch size (default 64)
    * SGD weight updates across filters, conv bias, fc weights, fc bias
    * Saves best model checkpoint by validation loss to ./models/
    * Resume training from checkpoint with --model flag
* Dataset cleaned of symbol/dingbat fonts that don't render standard digit glyphs (webdings, wingdings, symbol, etc.)

**Training Results So Far:**
* 1220 images across 10 classes (~122 per digit, ~130 fonts)
* Best val loss: 0.6675 | Best val acc: ~86% at epoch 30
* Training continuing from checkpoint

**Architecture:**
* Input: 28x28 grayscale image
* Conv layer: 8 filters, 3x3 kernel, stride 1, ReLU
* Max pooling: 2x2 window, stride 2
* Flatten → fully connected → softmax
* Output: 10 classes (digits 0-9)
