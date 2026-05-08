"""
train.py

Imports
Settings            - n_filters, filter_size, n_classes, batch_size, epochs, learning_rate, seed, model_dir
lock_random_state   - locks random state for reproducibility
load_images         - walks dataset folders, preprocesses each image into a (28,28) float32 array
label_to_vector     - converts an integer label into a class vector
shuffle_in_place    - randomly reorders images and labels in place
compute_accuracy    - compares argmax predictions to true labels
save_checkpoint     - compresses and writes model state to a .npz file
load_checkpoint     - loads a .npz checkpoint and returns parameters + training state
train               - main loop: init or resume, split data, run forward/backward propagation, validate, checkpoint
"""

import os
import sys
import glob
import random
import time
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(__file__))
import calculation_functions as cf
import pre_process

n_filters = 8
filter_size = 3
n_classes = 10
batch_size = 64
epochs = 100
learning_rate = 1e-3
seed = 42
model_dir = Path("./models")
model_dir.mkdir(parents=True, exist_ok=True)


def lock_random_state():
    random.seed(seed)
    np.random.seed(seed)


def load_images(root="./dataset"):
    images, labels = [], []
    for digit in range(10):
        for img_path in glob.glob(os.path.join(root, str(digit), "*")):
            try:
                img = pre_process.load_image(img_path)
                resized = pre_process.resize_image(img)
                images.append(pre_process.to_numpy(resized))
                labels.append(digit)
            except Exception as e:
                print(f"  skipping {img_path}: {e}")
    return images, labels


def label_to_vector(label):
    vec = np.zeros((n_classes,), dtype=np.float32)
    vec[label] = 1.0
    return vec


def shuffle_in_place(imgs, lbls):
    combined = list(zip(imgs, lbls))
    random.shuffle(combined)
    for i, (img, lbl) in enumerate(combined):
        imgs[i] = img
        lbls[i] = lbl


def compute_accuracy(probs, labels):
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == np.array(labels))


def save_checkpoint(state, path):
    np.savez_compressed(path, **state)


def load_checkpoint(path):
    data = np.load(path, allow_pickle=True)
    epoch = int(data["epoch"].item())
    val_loss = float(data["val_loss"].item())
    print(f"    Resumed from {path} (epoch {epoch}, val loss {val_loss:.4f})")
    return data["filter_bank"], data["conv_bias"], data["fc_weights"], data["fc_bias"], epoch, val_loss


def train(resume_path=None):
    lock_random_state()

    print("Loading dataset...")
    images, labels = load_images()
    total_images = len(images)
    print(f"    {total_images} images loaded")

    combined = list(zip(images, labels))
    random.shuffle(combined)
    images, labels = zip(*combined)
    images = list(images)
    labels = list(labels)

    conv_out_h = 28 - filter_size + 1
    conv_out_w = 28 - filter_size + 1
    pool_out_h = conv_out_h // 2
    pool_out_w = conv_out_w // 2
    flattened_size = n_filters * pool_out_h * pool_out_w

    start_epoch = 0
    best_val_loss = np.inf

    if resume_path:
        print("Resuming from checkpoint...")
        filters, conv_bias, fc_weights, fc_bias, start_epoch, best_val_loss = load_checkpoint(resume_path)
    else:
        print("Initialising parameters...")
        filters = cf.init_3d_filters(n_filters, filter_size).astype(np.float64)
        conv_bias = np.zeros(n_filters, dtype=np.float64)
        fc_weights = cf.init_weights(n_classes, flattened_size)
        fc_bias = np.zeros(n_classes, dtype=np.float64)

    train_split = int(0.8 * total_images)
    train_imgs = images[:train_split]
    val_imgs = images[train_split:]
    train_lbls = labels[:train_split]
    val_lbls = labels[train_split:]
    print(f"    Train: {len(train_imgs)} | Val: {len(val_imgs)}")

    steps_per_epoch = max(1, len(train_imgs) // batch_size)
    best_state = None

    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        epoch_start = time.time()
        shuffle_in_place(train_imgs, train_lbls)

        epoch_loss = 0.0
        epoch_acc = 0.0

        for step in range(steps_per_epoch):
            batch_imgs = train_imgs[step * batch_size : (step + 1) * batch_size]
            batch_lbls = train_lbls[step * batch_size : (step + 1) * batch_size]

            probs_batch, caches = [], []
            for img in batch_imgs:
                probs, cache = cf.forward_pass(img, filters, conv_bias, fc_weights, fc_bias)
                probs_batch.append(probs)
                caches.append(cache)
            probs_batch = np.stack(probs_batch)

            targets = np.stack([label_to_vector(l) for l in batch_lbls])
            epoch_loss += cf.cross_entropy(probs_batch, targets)
            epoch_acc += compute_accuracy(probs_batch, batch_lbls)

            grad_filters = np.zeros_like(filters)
            grad_conv_bias = np.zeros_like(conv_bias)
            grad_fc_weights = np.zeros_like(fc_weights)
            grad_fc_bias = np.zeros_like(fc_bias)

            for img, cache, targ in zip(batch_imgs, caches, targets):
                sample_grad_filters, sample_grad_conv_bias, sample_grad_fc_weights, sample_grad_fc_bias = \
                    cf.backward_pass(img, filters, conv_bias, fc_weights, fc_bias, targ, cache)
                grad_filters += sample_grad_filters
                grad_conv_bias += sample_grad_conv_bias
                grad_fc_weights += sample_grad_fc_weights
                grad_fc_bias += sample_grad_fc_bias

            filters -= learning_rate * grad_filters / batch_size
            conv_bias -= learning_rate * grad_conv_bias / batch_size
            fc_weights -= learning_rate * grad_fc_weights / batch_size
            fc_bias -= learning_rate * grad_fc_bias / batch_size

        val_probs = np.stack([
            cf.forward_pass(img, filters, conv_bias, fc_weights, fc_bias)[0]
            for img in val_imgs
        ])
        val_loss = cf.cross_entropy(val_probs, np.stack([label_to_vector(l) for l in val_lbls]))
        val_acc = compute_accuracy(val_probs, val_lbls)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {epoch_loss/steps_per_epoch:.4f} | train acc {epoch_acc/steps_per_epoch:.4f} | "
            f"val loss {val_loss:.4f} | val acc {val_acc:.4f} | "
            f"time {time.time() - epoch_start:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "filter_bank": filters.copy(),
                "conv_bias"  : conv_bias.copy(),
                "fc_weights" : fc_weights.copy(),
                "fc_bias"    : fc_bias.copy(),
                "epoch"      : epoch,
                "val_loss"   : float(val_loss),
                "val_acc"    : float(val_acc),
                "config"     : {
                    "n_filters"    : n_filters,
                    "filter_size"  : filter_size,
                    "n_classes"    : n_classes,
                    "batch_size"   : batch_size,
                    "learning_rate": learning_rate,
                },
            }
            ckpt = model_dir / f"best_model_epoch{epoch}.npz"
            save_checkpoint(best_state, ckpt)
            print(f"  > saved → {ckpt}")

    print("\nTraining finished.")
    print(f"Best val loss: {best_val_loss:.4f} (epoch {best_state['epoch']})")
    save_checkpoint(best_state, model_dir / "final_model.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    train(resume_path=args.model)
