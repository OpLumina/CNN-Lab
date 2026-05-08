#!/usr/bin/env python3
"""
train.py – end‑to‑end training for the tiny CNN.

* Loads images from ./dataset/<digit>/…
* Uses the preprocessing code from pre_process.py.
* Randomly initialises filters / dense layer.
* Trains with mini‑batches, prints loss/accuracy.
* Saves the best model (lowest validation loss) to ./models/.
"""

import os
import sys
import glob
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

# --------------------------------------------------------------
#  Local imports – do **not** modify the original modules
# --------------------------------------------------------------
sys.path.append(os.path.dirname(__file__))
import calculation_functions as cf
import pre_process                      # only the resize routine is needed

# --------------------------------------------------------------
#  Hyper‑parameters (feel free to tune)
# --------------------------------------------------------------
NUM_FILTERS   = 8
FILTER_SIZE   = 3
NUM_CLASSES   = 10
BATCH_SIZE    = 64
EPOCHS        = 30
LEARNING_RATE = 1e-3
SEED          = 42
MODEL_DIR     = Path("./models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------
#  Utility helpers (must appear **before** `train()`)
# --------------------------------------------------------------
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(root: str = "./dataset"):
    """Return (images, labels) where each image is a (28,28) float32 array."""
    images, labels = [], []
    for digit in range(10):
        pattern = os.path.join(root, str(digit), "*")
        for img_path in glob.glob(pattern):
            try:
                pil_img = Image.open(img_path).convert('1')
                processed = pre_process.resize_image_preserve_aspect_ratio(pil_img)
                arr = np.array(processed, dtype=np.float32) / 255.0  # (28,28)
                images.append(arr)
                labels.append(digit)
            except Exception as e:
                print(f"  [warn] skipping {img_path}: {e}")
    return images, labels


def one_hot(label: int, n: int = NUM_CLASSES):
    vec = np.zeros((n,), dtype=np.float32)
    vec[label] = 1.0
    return vec


def shuffle_data(imgs, lbls):
    combined = list(zip(imgs, lbls))
    random.shuffle(combined)
    for i, (img, lbl) in enumerate(combined):
        imgs[i] = img
        lbls[i] = lbl


def compute_accuracy(probs, true_labels):
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == np.array(true_labels))


def save_model(state: dict, path: Path):
    np.savez_compressed(path, **state)


# --------------------------------------------------------------
#  Training loop
# --------------------------------------------------------------
def train():
    set_seed()

    # ---------- 1 Load data ----------
    print("[1] Loading dataset …")
    images, labels = load_dataset()
    N = len(images)
    print(f"    Loaded {N} images across {NUM_CLASSES} classes.")

    # ---------- Shuffle BEFORE split so val set has all digits ----------
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images, labels = zip(*combined)
    images = list(images)
    labels = list(labels)

    # ---------- 2 Random initialisation ----------
    print("[2] Initialising parameters …")
    filter_bank = cf.init_3d_filters(NUM_FILTERS, FILTER_SIZE).astype(np.float64)
    conv_bias   = np.zeros(NUM_FILTERS, dtype=np.float64)

    conv_out_h = 28 - FILTER_SIZE + 1
    conv_out_w = 28 - FILTER_SIZE + 1
    pool_h     = conv_out_h // 2
    pool_w     = conv_out_w // 2
    flat_len   = NUM_FILTERS * pool_h * pool_w

    fc_weights = cf.init_weights(NUM_CLASSES, flat_len)   # (flat_len, C)
    fc_bias    = np.zeros(NUM_CLASSES, dtype=np.float64)

    # ---------- 3 Train / validate split ----------
    split_idx  = int(0.8 * N)
    train_imgs = images[:split_idx]
    val_imgs   = images[split_idx:]
    train_lbls = labels[:split_idx]
    val_lbls   = labels[split_idx:]
    print(f"    Train: {len(train_imgs)} | Val: {len(val_imgs)}")

    steps_per_epoch = max(1, len(train_imgs) // BATCH_SIZE)

    best_val_loss = np.inf
    best_state    = None

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        shuffle_data(train_imgs, train_lbls)

        epoch_loss = 0.0
        epoch_acc  = 0.0

        for step in range(steps_per_epoch):
            start = step * BATCH_SIZE
            end   = start + BATCH_SIZE
            batch_imgs = train_imgs[start:end]
            batch_lbls = train_lbls[start:end]

            # ----- forward pass -----
            probs_batch = []
            caches      = []
            for img in batch_imgs:
                probs, cache = cf.forward_pass(
                    img, filter_bank, conv_bias, fc_weights, fc_bias
                )
                probs_batch.append(probs)
                caches.append(cache)
            probs_batch = np.stack(probs_batch)   # (B, C)

            # ----- loss & accuracy -----
            targets = np.stack([one_hot(l) for l in batch_lbls])
            loss    = cf.cross_entropy(probs_batch, targets)
            acc     = compute_accuracy(probs_batch, batch_lbls)

            epoch_loss += loss
            epoch_acc  += acc

            # ----- backward pass -----
            dF_sum = np.zeros_like(filter_bank)
            dB_sum = np.zeros_like(conv_bias)
            dW_sum = np.zeros_like(fc_weights)
            db_sum = np.zeros_like(fc_bias)

            for img, cache, targ in zip(batch_imgs, caches, targets):
                dF, dB, dW, db = cf.backward_pass(
                    img, filter_bank, conv_bias, fc_weights, fc_bias,
                    targ, cache
                )
                dF_sum += dF
                dB_sum += dB
                dW_sum += dW
                db_sum += db

            # ----- SGD update -----
            filter_bank -= LEARNING_RATE * dF_sum / BATCH_SIZE
            conv_bias   -= LEARNING_RATE * dB_sum / BATCH_SIZE
            fc_weights  -= LEARNING_RATE * dW_sum / BATCH_SIZE
            fc_bias     -= LEARNING_RATE * db_sum / BATCH_SIZE

        # ---------- Validation ----------
        val_probs = []
        for img in val_imgs:
            p, _ = cf.forward_pass(img, filter_bank, conv_bias,
                                   fc_weights, fc_bias)
            val_probs.append(p)
        val_probs = np.stack(val_probs)
        val_loss  = cf.cross_entropy(
            val_probs,
            np.stack([one_hot(l) for l in val_lbls])
        )
        val_acc = compute_accuracy(val_probs, val_lbls)

        # ---------- Logging & checkpoint ----------
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {epoch_loss/steps_per_epoch:.4f} | "
            f"train acc {epoch_acc/steps_per_epoch:.4f} | "
            f"val loss {val_loss:.4f} | val acc {val_acc:.4f} | "
            f"time {epoch_time:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "filter_bank": filter_bank.copy(),
                "conv_bias"  : conv_bias.copy(),
                "fc_weights" : fc_weights.copy(),
                "fc_bias"    : fc_bias.copy(),
                "epoch"      : epoch,
                "val_loss"   : float(val_loss),
                "val_acc"    : float(val_acc),
                "config"     : {
                    "num_filters"  : NUM_FILTERS,
                    "filter_size"  : FILTER_SIZE,
                    "num_classes"  : NUM_CLASSES,
                    "batch_size"   : BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                },
            }
            ckpt_path = MODEL_DIR / f"best_model_epoch{epoch}.npz"
            save_model(best_state, ckpt_path)
            print(f"  > New best model saved → {ckpt_path}")

    # ---------- Final report ----------
    print("\nTraining finished.")
    print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_state['epoch']})")
    final_path = MODEL_DIR / "final_model.npz"
    save_model(best_state, final_path)
    print(f"Final model persisted to {final_path}")


# --------------------------------------------------------------
#  Entry point
# --------------------------------------------------------------
if __name__ == "__main__":
    train()