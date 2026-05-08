import numpy as np

# ----------------------------------------------------------------------
#  Initialise parameters
# ----------------------------------------------------------------------
def init_3d_filters(number, size):
    """
    Create *number* square filters of shape (size, size) and return a
    3‑D array with shape (number, size, size).
    """
    filters = []
    for i in range(number):
        current_filter = np.random.randint(-5, 5, size=(size, size))
        filters.append(current_filter)
    return np.array(filters)                # (number, size, size)


def init_weights(number, vector_length):
    """Small Gaussian initialise for a dense weight matrix."""
    return np.random.randn(vector_length, number) * 0.01


# ----------------------------------------------------------------------
#  Activation utilities
# ----------------------------------------------------------------------
def relu(x):
    return np.maximum(0, x)


def relu_backward(dL_dconv, pre_relu):
    drelu = (pre_relu > 0).astype(float)
    return dL_dconv * drelu


# ----------------------------------------------------------------------
#  Misc helpers
# ----------------------------------------------------------------------
def extract_array_from_array(x, y, width, height, array):
    """Extract a copy of a rectangular region from *array*."""
    return array[y:y + height, x:x + width].copy()


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def cross_entropy(pred_probs, target_one_hot):
    """
    pred_probs      : (batch, n_classes) – softmax output
    target_one_hot  : (batch, n_classes) – one‑hot ground truth
    Returns a scalar (average) loss.
    """
    eps = 1e-12
    log_preds = np.log(pred_probs + eps)
    loss = -np.mean(np.sum(target_one_hot * log_preds, axis=1))
    return loss


# ----------------------------------------------------------------------
#  Convolution / pooling
# ----------------------------------------------------------------------
def feature_map(filter_array, main_array, stride, bias):
    """
    Forward convolution + bias + ReLU.
    filter_array : (depth, k_h, k_w)
    main_array   : (H, W)
    bias         : (depth,)
    Returns      : (depth, out_h, out_w)
    """
    filter_depth, window_height, window_width = filter_array.shape
    main_array_height, main_array_width = main_array.shape

    horizontal_iterations_needed = ((main_array_width - window_width) // stride) + 1
    vertical_iterations_needed   = ((main_array_height - window_height) // stride) + 1

    four_d_feature_map_elements = []

    for y in range(0, main_array_height - window_height + 1, stride):
        for x in range(0, main_array_width - window_width + 1, stride):
            extracted_main_array = extract_array_from_array(
                x, y, window_width, window_height, main_array
            )
            # broadcasting ⇒ (depth, window_h, window_w)
            three_d_feature_map_element = np.multiply(extracted_main_array, filter_array)
            four_d_feature_map_elements.append(three_d_feature_map_element)

    four_d_feature_map_elements = np.array(four_d_feature_map_elements)

    five_d_features_matrixes = four_d_feature_map_elements.reshape(
        vertical_iterations_needed,
        horizontal_iterations_needed,
        filter_depth,
        window_height,
        window_width,
    )

    # Sum over the spatial dimensions → shape (vert, horiz, depth)
    feature_map = np.sum(five_d_features_matrixes, axis=(3, 4))
    feature_map = feature_map.transpose(2, 0, 1)   # (depth, out_h, out_w)

    # Add bias and apply ReLU
    feature_map += bias[:, None, None]
    pre_relu = feature_map.copy()          # cache before activation
    feature_map_final = np.maximum(0, feature_map)
    return feature_map_final, pre_relu



def conv_backward(dL_dconv_out, filter_bank, main_array, stride, bias):
    """
    dL_dconv_out : (n_filters, out_h, out_w) – gradient w.r.t. conv output (pre‑ReLU)
    filter_bank  : (n_filters, depth?, k_h, k_w) – same filters used in forward
    main_array   : (H, W) – original image
    bias         : (n_filters,)
    Returns:
        dL_dfilters : same shape as filter_bank (float)
        dL_dbias    : same shape as bias    (float)
        dL_dinput   : (H, W)                (float)
    """
    # --------------------------------------------------------------
    #  Accept both (n, k, k) and (n, depth, k, k) shapes
    # --------------------------------------------------------------
    shape = filter_bank.shape
    if len(shape) == 3:                     # (n_filters, k_h, k_w)
        n_filters, k_h, k_w = shape
        depth = 1
        filter_bank = filter_bank[:, None, :, :]   # -> (n, 1, k_h, k_w)
    else:                                   # (n_filters, depth, k_h, k_w)
        n_filters, depth, k_h, k_w = shape

    H, W = main_array.shape
    out_h, out_w = dL_dconv_out.shape[1:]

    # ------------------------------------------------------------------
    #  Use floating‑point buffers – avoid int‑overflow / casting errors
    # ------------------------------------------------------------------
    dL_dfilters = np.zeros_like(filter_bank, dtype=np.float64)
    dL_dbias    = np.zeros_like(bias, dtype=np.float64)
    dL_dinput   = np.zeros_like(main_array, dtype=np.float64)

    for f in range(n_filters):
        for oy in range(out_h):
            for ox in range(out_w):
                y = oy * stride
                x = ox * stride
                patch = extract_array_from_array(x, y, k_w, k_h, main_array)
                grad = dL_dconv_out[f, oy, ox]                # scalar float

                # Gradient w.r.t. each filter channel
                dL_dfilters[f] += grad * patch
                dL_dbias[f]    += grad
                # Accumulate gradient on the input (sum over depth if depth>1)
                dL_dinput[y:y + k_h, x:x + k_w] += grad * filter_bank[f].sum(axis=0)

    # If we added a dummy depth dimension, squeeze it back before returning
    if depth == 1:
        dL_dfilters = dL_dfilters.squeeze(axis=1)   # shape (n_filters, k_h, k_w)

    return dL_dfilters, dL_dbias, dL_dinput



def maxpool(main_array, window_size=2, stride=2):
    """2‑D max‑pooling over a tensor of shape (depth, H, W)."""
    depth, h, w = main_array.shape
    horizontal_iterations_needed = (w - window_size) // stride + 1
    vertical_iterations_needed   = (h - window_size) // stride + 1

    pooled_array = np.zeros((depth, vertical_iterations_needed,
                             horizontal_iterations_needed))

    for d in range(depth):
        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                window = main_array[d, y:y + window_size, x:x + window_size]
                pooled_array[d, y // stride, x // stride] = np.max(window)

    return pooled_array


def maxpool_backward(dL_dpooled, pooled, maxpooled_input):
    """
    dL_dpooled      : (depth, h_out, w_out) – upstream gradient
    pooled          : (depth, h_out, w_out) – result of maxpool (stored)
    maxpooled_input : (depth, h_in, w_in)   – tensor that entered maxpool
    Returns gradient w.r.t. the input of maxpool.
    """
    depth, h_in, w_in = maxpooled_input.shape
    _, h_out, w_out   = dL_dpooled.shape
    window = 2          # hard‑coded 2×2 window
    stride = 2

    dL_dinput = np.zeros_like(maxpooled_input)

    for d in range(depth):
        for i in range(h_out):
            for j in range(w_out):
                y_start = i * stride
                x_start = j * stride
                window_slice = maxpooled_input[d,
                                              y_start:y_start + window,
                                              x_start:x_start + window]
                max_val = np.max(window_slice)
                mask = (window_slice == max_val).astype(float)
                dL_dinput[d,
                          y_start:y_start + window,
                          x_start:x_start + window] += mask * dL_dpooled[d, i, j]

    return dL_dinput


# ----------------------------------------------------------------------
#  Fully‑connected utilities
# ----------------------------------------------------------------------
def flatten(array):
    return array.flatten()


def dloss_dlogits(pred_probs, target_one_hot):
    """∂L/∂z for softmax + cross‑entropy."""
    return pred_probs - target_one_hot.squeeze()


def linear_backward(dL_dlogits, flat, fc_W):
    """
    dL_dlogits : (n_classes,)
    flat       : (flattened_feature_len,)
    fc_W       : (flattened_feature_len, n_classes)
    Returns:
        dL_dW   : (flattened_feature_len, n_classes)
        dL_db   : (n_classes,)
        dL_dflat: (flattened_feature_len,)
    """
    dL_dW   = np.outer(flat, dL_dlogits)
    dL_db   = dL_dlogits.copy()
    dL_dflat = fc_W @ dL_dlogits
    return dL_dW, dL_db, dL_dflat


def unflatten(dL_dflat, pooled_shape):
    """Reshape flattened gradient back to the pooled tensor shape."""
    return dL_dflat.reshape(pooled_shape)


# ----------------------------------------------------------------------
#  End‑to‑end forward / backward passes
# ----------------------------------------------------------------------
def forward_pass(image, filter_bank, bias, fc_weights, fc_bias):
    conv_out,pre_relu = feature_map(filter_bank, image, stride=1, bias=bias)   # (n_filters, out_h, out_w)
    pooled   = maxpool(conv_out)                                      # (n_filters, out_h/2, out_w/2)
    flat     = flatten(pooled)                                        # (flattened_feature_len,)
    logits   = np.dot(flat, fc_weights) + fc_bias                     # (n_classes,)
    probs    = softmax(logits)
    cache = (conv_out, pre_relu, pooled, flat, logits)
    return probs, cache


def backward_pass(image, filter_bank, bias, fc_weights, fc_bias,
                  target_one_hot, cache):
    conv_out, pre_relu, pooled, flat, logits = cache
    dL_dlogits = dloss_dlogits(softmax(logits), target_one_hot)      # (n_classes,)
    dL_dW, dL_db, dL_dflat = linear_backward(dL_dlogits, flat, fc_weights)
    dL_dpooled = unflatten(dL_dflat, pooled.shape)                  # (depth, h_out, w_out)
    dL_dconv_pre_relu = maxpool_backward(dL_dpooled, pooled, conv_out)
    dL_dconv = relu_backward(dL_dconv_pre_relu, pre_relu)
    dL_dfilters, dL_dbias, _ = conv_backward(dL_dconv, filter_bank,
                                             image, stride=1, bias=bias)
    return dL_dfilters, dL_dbias, dL_dW, dL_db
