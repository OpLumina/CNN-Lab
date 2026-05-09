import numpy as np


# Parameter initialisation

def init_3d_filters(number, size):
    filters = []
    for i in range(number):
        current_filter = np.random.randint(-5, 5, size=(size, size))
        filters.append(current_filter)
    return np.array(filters)

def init_weights(number, vector_length):
    return np.random.randn(vector_length, number) * 0.01


# Activations

def relu(x):
    return np.maximum(0, x)

def relu_backward(upstream_grad, pre_relu):
    activation_mask = (pre_relu > 0).astype(float)
    return upstream_grad * activation_mask


# Misc helpers

def extract_array_from_array(x, y, width, height, array):
    return array[y:y + height, x:x + width].copy()

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def cross_entropy(pred_probs, target_one_hot):
    eps = 1e-12
    log_preds = np.log(pred_probs + eps)
    loss = -np.mean(np.sum(target_one_hot * log_preds, axis=1))
    return loss


# Convolution and pooling

def feature_map(filter_array, main_array, stride, bias):
    filter_depth, window_height, window_width = filter_array.shape
    main_array_height, main_array_width = main_array.shape

    horizontal_iterations_needed = ((main_array_width - window_width) // stride) + 1
    vertical_iterations_needed = ((main_array_height - window_height) // stride) + 1

    four_d_feature_map_elements = []

    for y in range(0, main_array_height - window_height + 1, stride):
        for x in range(0, main_array_width - window_width + 1, stride):
            extracted_main_array = extract_array_from_array(x, y, window_width, window_height, main_array)
            three_d_feature_map_element = np.multiply(extracted_main_array, filter_array)
            four_d_feature_map_elements.append(three_d_feature_map_element)

    # This part took me like 8 hours and like 20 pieces of graph paper to make sense of, 
    # I should've listened in Algebra II
    four_d_feature_map_elements = np.array(four_d_feature_map_elements)

    five_d_features_matrixes = four_d_feature_map_elements.reshape(
        vertical_iterations_needed, horizontal_iterations_needed, filter_depth, window_height, window_width,
    )

    feature_map = np.sum(five_d_features_matrixes, axis=(3, 4))
    feature_map = feature_map.transpose(2, 0, 1)

    feature_map += bias[:, None, None]
    pre_relu = feature_map.copy()
    feature_map_final = np.maximum(0, feature_map)
    return feature_map_final, pre_relu


def conv_backward(upstream_grad, filter_bank, main_array, stride, bias):
    shape = filter_bank.shape
    if len(shape) == 3:
        n_filters, k_h, k_w = shape
        depth = 1
        filter_bank = filter_bank[:, None, :, :]
    else:
        n_filters, depth, k_h, k_w = shape

    H, W = main_array.shape
    out_h, out_w = upstream_grad.shape[1:]

    filter_gradients = np.zeros_like(filter_bank, dtype=np.float64)
    bias_gradients = np.zeros_like(bias, dtype=np.float64)
    input_gradients = np.zeros_like(main_array, dtype=np.float64)

    for filter_idx in range(n_filters):
        for out_row in range(out_h):
            for out_col in range(out_w):
                y = out_row * stride
                x = out_col * stride
                input_patch = extract_array_from_array(x, y, k_w, k_h, main_array)
                local_grad = upstream_grad[filter_idx, out_row, out_col]
                filter_gradients[filter_idx] += local_grad * input_patch
                bias_gradients[filter_idx] += local_grad
                input_gradients[y:y + k_h, x:x + k_w] += local_grad * filter_bank[filter_idx].sum(axis=0)

    if depth == 1:
        filter_gradients = filter_gradients.squeeze(axis=1)

    return filter_gradients, bias_gradients, input_gradients


def maxpool(main_array, window_size=2, stride=2):
    depth, h, w = main_array.shape
    horizontal_iterations_needed = (w - window_size) // stride + 1
    vertical_iterations_needed = (h - window_size) // stride + 1

    pooled_array = np.zeros((depth, vertical_iterations_needed, horizontal_iterations_needed))

    for d in range(depth):
        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                window = main_array[d, y:y + window_size, x:x + window_size]
                pooled_array[d, y // stride, x // stride] = np.max(window)

    return pooled_array


def maxpool_backward(upstream_grad, pooled, maxpooled_input):
    depth, h_in, w_in = maxpooled_input.shape
    _, h_out, w_out = upstream_grad.shape
    window = 2
    stride = 2

    input_gradients = np.zeros_like(maxpooled_input)

    for d in range(depth):
        for row in range(h_out):
            for col in range(w_out):
                y_start = row * stride
                x_start = col * stride
                input_window = maxpooled_input[d, y_start:y_start + window, x_start:x_start + window]
                max_val = np.max(input_window)
                mask = (input_window == max_val).astype(float)
                input_gradients[d, y_start:y_start + window, x_start:x_start + window] += mask * upstream_grad[d, row, col]

    return input_gradients


# Fully connected layer

def flatten(array):
    return array.flatten()

# Gradient loss before softmax
def dloss_dlogits(pred_probs, target_one_hot):
    return pred_probs - target_one_hot.squeeze()

def linear_backward(upstream_grad, flat, fc_W):
    weight_gradients = np.outer(flat, upstream_grad)
    bias_gradients = upstream_grad.copy()
    input_gradients = fc_W @ upstream_grad
    return weight_gradients, bias_gradients, input_gradients

def unflatten(input_gradients, pooled_shape):
    return input_gradients.reshape(pooled_shape)


# Forward and backward passes

def forward_pass(image, filter_bank, bias, fc_weights, fc_bias):
    conv_out, pre_relu = feature_map(filter_bank, image, stride=1, bias=bias)
    pooled = maxpool(conv_out)
    flat = flatten(pooled)
    logits = np.dot(flat, fc_weights) + fc_bias
    probs = softmax(logits)
    cache = (conv_out, pre_relu, pooled, flat, logits)
    return probs, cache

def backward_pass(image, filter_bank, bias, fc_weights, fc_bias, target_one_hot, cache):
    conv_out, pre_relu, pooled, flat, logits = cache
    output_grad = dloss_dlogits(softmax(logits), target_one_hot)
    fc_weight_grad, fc_bias_grad, flat_grad = linear_backward(output_grad, flat, fc_weights)
    pooled_grad = unflatten(flat_grad, pooled.shape)
    pre_relu_grad = maxpool_backward(pooled_grad, pooled, conv_out)
    conv_grad = relu_backward(pre_relu_grad, pre_relu)
    filter_grad, conv_bias_grad, _ = conv_backward(conv_grad, filter_bank, image, stride=1, bias=bias)
    return filter_grad, conv_bias_grad, fc_weight_grad, fc_bias_grad
