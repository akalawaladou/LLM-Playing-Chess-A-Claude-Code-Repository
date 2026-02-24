"""
Training pipeline for OthelloGPT.

Steps:
    1. Generate N random Othello games -> sequences of move tokens
    2. Pad/batch the sequences
    3. Train the GPT with a standard next-token cross-entropy loss
    4. Save the trained model + dataset (boards) for probing

Usage:
    python train.py              # trains with default settings
    python train.py --n_games 50000 --epochs 10
"""

import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from othello import generate_dataset
from model import OthelloGPT


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
PAD_TOKEN = 64  # vocab 0..63 are board squares, 64 = padding


class OthelloDataset(Dataset):
    """
    Each sample is one Othello game represented as:
        input:  [m0, m1, ..., m_{T-2}]   (moves 0 to T-2)
        target: [m1, m2, ..., m_{T-1}]   (shifted by 1)
    """

    def __init__(self, games: list, max_len: int = 64):
        self.samples = []
        for moves in games:
            t = torch.tensor(moves, dtype=torch.long)
            # Pad to max_len
            if len(t) < max_len:
                pad = torch.full((max_len - len(t),), PAD_TOKEN, dtype=torch.long)
                t = torch.cat([t, pad])
            else:
                t = t[:max_len]
            self.samples.append(t)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        return seq[:-1], seq[1:]  # input, target


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(
    n_games: int = 20_000,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 3e-4,
    n_layers: int = 8,
    d_model: int = 128,
    n_heads: int = 4,
    device: str = "cpu",
    save_dir: str = "checkpoints",
):
    print(f"=== Generating {n_games} random Othello games ===")
    all_moves, all_boards = generate_dataset(n_games, seed=42)
    print(f"Generated {len(all_moves)} games "
          f"(avg length {np.mean([len(m) for m in all_moves]):.1f} moves)")

    # Save boards for later probing
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "dataset.pkl"), "wb") as f:
        pickle.dump({"moves": all_moves, "boards": all_boards}, f)
    print(f"Saved dataset to {save_dir}/dataset.pkl")

    # Split: 90% train, 10% val
    split = int(0.9 * len(all_moves))
    train_ds = OthelloDataset(all_moves[:split])
    val_ds = OthelloDataset(all_moves[split:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model
    model = OthelloGPT(
        n_layers=n_layers, d_model=d_model, n_heads=n_heads,
        max_len=64, vocab_size=65, dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters ({n_layers} layers, d={d_model})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    # --- Training ---
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            out = model(inp)
            logits = out["logits"]  # (B, T, V)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                logits = model(inp)["logits"]
                val_loss += criterion(
                    logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)
                ).item()

                # Top-1 accuracy (ignoring padding)
                preds = logits.argmax(dim=-1)
                mask = tgt != PAD_TOKEN
                correct += (preds[mask] == tgt[mask]).sum().item()
                total += mask.sum().item()

        dt = time.time() - t0
        print(f"Epoch {epoch:2d}/{epochs} | "
              f"train_loss {total_loss / len(train_loader):.4f} | "
              f"val_loss {val_loss / len(val_loader):.4f} | "
              f"val_acc {correct / total:.3f} | "
              f"{dt:.1f}s")

    # Save model
    ckpt_path = os.path.join(save_dir, "othello_gpt.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nModel saved to {ckpt_path}")
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train(
        n_games=args.n_games,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        device=device,
    )
