"""
train.py
--------
Training loop for predictive maintenance models.

Supports:
- Autoencoder (unsupervised, trained on healthy data only)
- HybridFaultDetector (LSTM + Transformer, binary classification)
- RULEstimator (regression, optional backbone fine-tuning)

Features:
- Early stopping
- Model checkpointing
- Learning rate scheduling
- Training metrics logging

Usage:
    python src/train.py --model autoencoder
    python src/train.py --model hybrid --epochs 50 --batch-size 64
    python src/train.py --model rul --freeze-backbone
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
import argparse
import os
import json
from datetime import datetime
from typing import Optional

from models.lstm_transformer import HybridFaultDetector, ConvAutoencoder, RULEstimator, AsymmetricRULLoss


# ─── Device ───────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_sequences(data_path: str = "data/processed") -> dict:
    """
    Load preprocessed windowed sequences from disk.

    Expected files:
        X_train.npy, y_train.npy, rul_train.npy
        X_val.npy,   y_val.npy,   rul_val.npy
        X_test.npy,  y_test.npy,  rul_test.npy

    Returns:
        Dict with 'train', 'val', 'test' keys, each containing X, y, rul arrays.
    """
    splits = {}
    for split in ["train", "val", "test"]:
        X = np.load(os.path.join(data_path, f"X_{split}.npy"))
        y = np.load(os.path.join(data_path, f"y_{split}.npy"))
        rul = np.load(os.path.join(data_path, f"rul_{split}.npy"))
        splits[split] = {"X": X, "y": y, "rul": rul}
        print(f"  {split}: X={X.shape}, fault_rate={y.mean():.1%}")
    return splits


def make_dataloaders(
    splits: dict,
    batch_size: int = 64,
    model_type: str = "hybrid"
) -> dict:
    """Build DataLoader objects for each split."""
    loaders = {}

    for split, data in splits.items():
        X = torch.tensor(data["X"], dtype=torch.float32)
        y = torch.tensor(data["y"], dtype=torch.float32).unsqueeze(1)
        rul = torch.tensor(data["rul"], dtype=torch.float32).unsqueeze(1)

        if model_type == "autoencoder":
            # Train only on healthy sequences
            if split == "train":
                mask = data["y"] == 0
                X = X[mask]
                y = y[mask]
                rul = rul[mask]

        dataset = TensorDataset(X, y, rul)

        # Class-balanced sampling for classification (train only)
        sampler = None
        if split == "train" and model_type == "hybrid":
            labels = data["y"]
            class_counts = np.bincount(labels.astype(int))
            weights = 1.0 / class_counts[labels.astype(int)]
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(split == "train" and sampler is None),
            num_workers=0,
            pin_memory=DEVICE.type == "cuda"
        )

    return loaders


# ─── Early Stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        improved = (
            metric < self.best - self.min_delta if self.mode == "min"
            else metric > self.best + self.min_delta
        )
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ─── Training Loops ───────────────────────────────────────────────────────────

def train_hybrid(
    model: HybridFaultDetector,
    loaders: dict,
    epochs: int = 50,
    lr: float = 1e-3,
    checkpoint_dir: str = "models"
) -> dict:
    """
    Train HybridFaultDetector for binary fault classification.

    Returns:
        Training history dict.
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()
    early_stopper = EarlyStopping(patience=10)

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_losses = []
        for X_batch, y_batch, _ in loaders["train"]:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validate ──
        model.eval()
        val_losses, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for X_batch, y_batch, _ in loaders["val"]:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_losses.append(loss.item())
                preds = (torch.sigmoid(logits) > 0.5).float()
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(y_batch.cpu().numpy().flatten())

        # Compute F1
        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)
        tp = ((preds_arr == 1) & (labels_arr == 1)).sum()
        fp = ((preds_arr == 1) & (labels_arr == 0)).sum()
        fn = ((preds_arr == 0) & (labels_arr == 1)).sum()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(f1)

        scheduler.step()

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_F1={f1:.4f}")

        # Checkpoint best model
        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "hybrid_best.pt"))
            print(f"  ✅ New best F1={best_f1:.4f} — checkpoint saved")

        if early_stopper(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nTraining complete. Best val F1: {best_f1:.4f}")
    return history


def train_autoencoder(
    model: ConvAutoencoder,
    loaders: dict,
    epochs: int = 40,
    lr: float = 5e-4,
    checkpoint_dir: str = "models"
) -> dict:
    """Train Autoencoder on healthy sequences only."""
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    early_stopper = EarlyStopping(patience=8)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X_batch, _, _ in loaders["train"]:
            X_batch = X_batch.to(DEVICE)
            optimizer.zero_grad()
            recon, _ = model(X_batch)
            loss = criterion(recon, X_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, _, _ in loaders["val"]:
                X_batch = X_batch.to(DEVICE)
                recon, _ = model(X_batch)
                val_losses.append(criterion(recon, X_batch).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss == min(history["val_loss"]):
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "autoencoder_best.pt"))

        if early_stopper(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    return history


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="hybrid",
                        choices=["hybrid", "autoencoder", "rul"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--checkpoint-dir", type=str, default="models")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Device: {DEVICE}")
    print(f"Model: {args.model} | Epochs: {args.epochs} | Batch: {args.batch_size}")
    print("-" * 60)

    splits = load_sequences(args.data_dir)
    loaders = make_dataloaders(splits, args.batch_size, args.model)

    if args.model == "hybrid":
        model = HybridFaultDetector(n_sensors=8)
        history = train_hybrid(model, loaders, args.epochs, args.lr, args.checkpoint_dir)

    elif args.model == "autoencoder":
        model = ConvAutoencoder(n_sensors=8, seq_len=50)
        history = train_autoencoder(model, loaders, args.epochs, args.lr, args.checkpoint_dir)

    elif args.model == "rul":
        backbone = HybridFaultDetector(n_sensors=8)
        ckpt = os.path.join(args.checkpoint_dir, "hybrid_best.pt")
        if os.path.exists(ckpt):
            backbone.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            print(f"Loaded backbone from {ckpt}")
        model = RULEstimator(backbone, freeze_backbone=args.freeze_backbone)
        history = train_hybrid(model, loaders, args.epochs, args.lr, args.checkpoint_dir)

    # Save history
    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/{args.model}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved → outputs/{args.model}_history.json")
