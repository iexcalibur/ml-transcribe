#!/usr/bin/env python3
"""Train a small MLP on MNIST and export fixtures for the Swift test suite.

Architecture matches ``swift/Sources/Transcribe/MLP.swift``:

    x (784) -> Linear -> ReLU -> Linear -> logits (10)

Weight layout conventions (different from PyTorch's defaults):

    "fc1.weight"   shape [784, 128]   (PyTorch stores [128, 784] -> transpose)
    "fc1.bias"     shape [1,   128]   (PyTorch stores [128]      -> unsqueeze)
    "fc2.weight"   shape [128,  10]
    "fc2.bias"     shape [1,    10]

Outputs:

    swift/Tests/Fixtures/mnist_mlp.safetensors   trained weights (F32)
    swift/Tests/Fixtures/mnist_samples.json      one flattened test image
                                                  per digit (0-9) plus labels

Run:

    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install torch torchvision safetensors
    python scripts/train_mnist.py

Deterministic seeding gives the same weights each run, so Swift tests
can assume a known accuracy threshold.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / ".data"
FIXTURE_DIR = REPO_ROOT / "swift" / "Tests" / "TranscribeTests" / "Fixtures"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
HIDDEN = 128
EPOCHS = 5
BATCH_SIZE = 128
LR = 1e-3
SEED = 42


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def make_loaders() -> tuple[DataLoader, DataLoader, datasets.MNIST]:
    # Normalize with the standard MNIST mean/std so the Swift side gets
    # pixel values in the same distribution the model was trained on.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(
        root=str(DATA_DIR), train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        root=str(DATA_DIR), train=False, download=True, transform=transform
    )
    train = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train, test, test_ds


def train(model: MLP, train_loader: DataLoader, test_loader: DataLoader) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        model.train()
        for images, labels in train_loader:
            images = images.view(-1, 784).to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(-1, 784).to(DEVICE)
                predicted = model(images).argmax(dim=1).cpu()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"epoch {epoch + 1}/{EPOCHS}  test_acc={acc:.4f}")
    return acc


def export_weights(model: MLP, path: Path) -> None:
    """Write weights to safetensors using the [in, out] convention.

    PyTorch's nn.Linear stores weights as [out, in] and evaluates
    `y = x @ W.T + b`. Our Swift MLP2 uses [in, out] and evaluates
    `y = x @ W + b`. Transposing on export makes the two equivalent.
    Bias is unsqueezed to [1, out] to match our op's shape expectations.
    """
    weights = {
        "fc1.weight": model.fc1.weight.detach().t().contiguous(),  # [784, 128]
        "fc1.bias":   model.fc1.bias.detach().unsqueeze(0).contiguous(),  # [1, 128]
        "fc2.weight": model.fc2.weight.detach().t().contiguous(),  # [128, 10]
        "fc2.bias":   model.fc2.bias.detach().unsqueeze(0).contiguous(),  # [1, 10]
    }
    save_file(weights, str(path))
    print(f"wrote {path}")
    for name, tensor in weights.items():
        print(f"    {name}: shape={list(tensor.shape)} dtype={tensor.dtype}")


def export_samples(test_ds: datasets.MNIST, path: Path, n_per_class: int = 1) -> None:
    """Pick a few representative test images and save their flattened,
    normalized pixel values along with the ground-truth label.

    Saving one example per class gives Swift a tiny but complete test
    set — every digit 0..9 is represented.
    """
    per_class: dict[int, list[float]] = {}
    expected_predictions: list[dict] = []
    for image, label in test_ds:
        if label in per_class:
            continue
        per_class[label] = image.view(-1).tolist()
        expected_predictions.append({"label": int(label), "pixels": per_class[label]})
        if len(per_class) >= 10 * n_per_class:
            break

    # Sort by label for deterministic ordering.
    expected_predictions.sort(key=lambda s: s["label"])
    with open(path, "w") as f:
        json.dump(expected_predictions, f)
    print(f"wrote {path} ({len(expected_predictions)} samples, 1 per class)")


def main() -> int:
    torch.manual_seed(SEED)
    train_loader, test_loader, test_ds = make_loaders()
    model = MLP().to(DEVICE)
    final_acc = train(model, train_loader, test_loader)

    weights_path = FIXTURE_DIR / "mnist_mlp.safetensors"
    samples_path = FIXTURE_DIR / "mnist_samples.json"
    export_weights(model, weights_path)
    export_samples(test_ds, samples_path)

    print()
    print(f"final test acc: {final_acc:.4f}")
    print(f"fixtures ready: {weights_path.name}, {samples_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
