"""Predict clean class IDs from raw uint8 .npy images with a trusted checkpoint."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import preprocess_images
from .models import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["dataset"], checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    images = np.load(args.images, allow_pickle=False)
    if not len(images):
        parser.error("Empty image array")
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(images), args.batch_size):
            batch = torch.from_numpy(preprocess_images(images[start:start + args.batch_size], checkpoint["dataset"]))
            probabilities.append(model(batch).softmax(1).numpy())
    probabilities = np.concatenate(probabilities)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, predictions=probabilities.argmax(1), probabilities=probabilities)


if __name__ == "__main__":
    main()
