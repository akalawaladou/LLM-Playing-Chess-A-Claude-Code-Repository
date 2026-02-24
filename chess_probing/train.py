"""
Training pipeline for ChessGPT.

Steps:
    1. Generate N random chess games -> sequences of UCI move tokens
    2. Build vocabulary (move string -> integer)
    3. Pad/batch the sequences
    4. Train the GPT with standard next-token cross-entropy loss
    5. Save model + dataset + vocabulary for probing

Usage:
    python train.py                     # default settings
    python train.py --n_games 10000 --epochs 15
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from chess_engine import generate_dataset, build_vocab, save_vocab
from model import ChessGPT


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChessDataset(Dataset):
    """
    Each sample is one chess game as:
        input:  [m0, m1, ..., m_{T-2}]   (moves 0 to T-2)
        target: [m1, m2, ..., m_{T-1}]   (shifted by 1)
    Moves are converted from UCI strings to integer tokens via the vocab.
    PAD_TOKEN = vocab_size  (one beyond the last real move token).
    """

    def __init__(self, games: list, vocab: dict, max_len: int = 80):
        self.pad = len(vocab)  # PAD_TOKEN
        self.samples = []
        for moves in games:
            tokens = [vocab[m] for m in moves if m in vocab]
            t = torch.tensor(tokens, dtype=torch.long)
            if len(t) < max_len:
                pad = torch.full((max_len - len(t),), self.pad, dtype=torch.long)
                t = torch.cat([t, pad])
            else:
                t = t[:max_len]
            self.samples.append(t)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        return seq[:-1], seq[1:]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    n_games: int = 10_000,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 3e-4,
    n_layers: int = 8,
    d_model: int = 128,
    n_heads: int = 4,
    max_len: int = 80,
    device: str = "cpu",
    save_dir: str = "checkpoints",
):
    print(f"=== Generating {n_games} random chess games ===")
    all_moves, all_boards = generate_dataset(n_games, seed=42)
    print(f"Generated {len(all_moves)} games "
          f"(avg length {np.mean([len(m) for m in all_moves]):.1f} moves)")

    # Build vocabulary from all training games
    vocab = build_vocab(all_moves)
    vocab_size = len(vocab)
    pad_token = vocab_size  # one beyond last token
    print(f"Vocabulary: {vocab_size} unique UCI moves")

    # Save everything
    os.makedirs(save_dir, exist_ok=True)
    save_vocab(vocab, os.path.join(save_dir, "vocab.json"))
    with open(os.path.join(save_dir, "chess_dataset.pkl"), "wb") as f:
        pickle.dump({"moves": all_moves, "boards": all_boards, "vocab": vocab}, f)
    print(f"Saved vocab + dataset to {save_dir}/")

    # 90/10 split
    split = int(0.9 * len(all_moves))
    train_ds = ChessDataset(all_moves[:split], vocab, max_len)
    val_ds   = ChessDataset(all_moves[split:], vocab, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    # Model — vocab_size + 1 for padding token
    model = ChessGPT(
        n_layers=n_layers, d_model=d_model, n_heads=n_heads,
        max_len=max_len, vocab_size=vocab_size + 1, dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"ChessGPT: {n_params:,} parameters ({n_layers} layers, d={d_model})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            logits = model(inp)["logits"]
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = total = 0
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                logits = model(inp)["logits"]
                val_loss += criterion(
                    logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)
                ).item()
                preds = logits.argmax(dim=-1)
                mask = tgt != pad_token
                correct += (preds[mask] == tgt[mask]).sum().item()
                total += mask.sum().item()

        dt = time.time() - t0
        print(f"Epoch {epoch:2d}/{epochs} | "
              f"train_loss {total_loss / len(train_loader):.4f} | "
              f"val_loss {val_loss / len(val_loader):.4f} | "
              f"val_acc {correct / max(total, 1):.3f} | "
              f"{dt:.1f}s")

    ckpt = os.path.join(save_dir, "chess_gpt.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"\nModel saved to {ckpt}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games", type=int, default=10_000)
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
