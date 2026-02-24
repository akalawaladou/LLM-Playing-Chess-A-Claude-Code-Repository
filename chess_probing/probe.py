"""
Probing analysis for ChessGPT — 13-class per-square classification.

Board classes:
    0  = empty
    1-6  = white Pawn/Knight/Bishop/Rook/Queen/King
    7-12 = black Pawn/Knight/Bishop/Rook/Queen/King

Same methodology as othello_probing/probe.py, extended to 13 classes.
The probe's task is harder than Othello (3 classes) but more informative:
it must recover not just *who* occupies a square but *which piece type*.

Usage:
    python probe.py
    python probe.py --layer 6
"""

import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from chess_engine import load_vocab
from model import ChessGPT


N_CLASSES = 13   # 0=empty, 1-6 white, 7-12 black


# ---------------------------------------------------------------------------
# Probe architectures (same as Othello, just n_classes=13)
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, d_model: int, n_squares: int = 64, n_classes: int = N_CLASSES):
        super().__init__()
        self.linear = nn.Linear(d_model, n_squares * n_classes)
        self.n_squares = n_squares
        self.n_classes = n_classes

    def forward(self, x):
        return self.linear(x).view(-1, self.n_squares, self.n_classes)


class NonLinearProbe(nn.Module):
    def __init__(self, d_model: int, hidden: int = 256,
                 n_squares: int = 64, n_classes: int = N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_squares * n_classes),
        )
        self.n_squares = n_squares
        self.n_classes = n_classes

    def forward(self, x):
        return self.net(x).view(-1, self.n_squares, self.n_classes)


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def extract_activations(
    model: ChessGPT,
    games_moves: list,
    games_boards: list,
    vocab: dict,
    max_samples: int = 8_000,
    device: str = "cpu",
    max_len: int = 80,
) -> dict:
    """
    Pass games through ChessGPT and collect (activation, board_state) pairs.

    Returns:
        "activations": {layer_idx: np.array (N, d_model)}
        "boards":      np.array (N, 8, 8)  values in 0..12
    """
    model.eval()
    layer_acts = {i: [] for i in range(model.n_layers)}
    board_labels = []
    n_collected = 0

    with torch.no_grad():
        for moves, boards in zip(games_moves, games_boards):
            if n_collected >= max_samples:
                break
            tokens = [vocab[m] for m in moves if m in vocab]
            if len(tokens) < 2:
                continue
            tokens = tokens[:max_len - 1]
            seq = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            out = model(seq, return_activations=True)

            for t in range(len(tokens)):
                if n_collected >= max_samples:
                    break
                if t >= len(boards):
                    break
                for layer_idx, act_tensor in out["activations"].items():
                    layer_acts[layer_idx].append(act_tensor[0, t].cpu().numpy())
                board_labels.append(boards[t].copy())
                n_collected += 1

    result = {
        "activations": {k: np.stack(v) for k, v in layer_acts.items()},
        "boards": np.stack(board_labels),
    }
    print(f"Extracted {n_collected} (position, activation) pairs")
    return result


# ---------------------------------------------------------------------------
# Probe dataset
# ---------------------------------------------------------------------------

class ProbeDataset(Dataset):
    def __init__(self, activations: np.ndarray, boards: np.ndarray):
        self.X = torch.tensor(activations, dtype=torch.float32)
        # boards already in 0..12, just flatten to 64
        self.Y = torch.tensor(boards.reshape(-1, 64), dtype=torch.long)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]


def train_probe(
    probe: nn.Module,
    train_acts, train_boards, val_acts, val_boards,
    epochs: int = 20, lr: float = 1e-3, batch_size: int = 256,
    device: str = "cpu",
) -> dict:
    probe = probe.to(device)
    train_loader = DataLoader(ProbeDataset(train_acts, train_boards),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(ProbeDataset(val_acts, val_boards),
                              batch_size=batch_size)

    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val = 0.0

    for epoch in range(1, epochs + 1):
        probe.train()
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            loss = criterion(probe(X).reshape(-1, N_CLASSES), Y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()

        probe.eval()
        correct = total = 0
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                preds = probe(X).argmax(dim=-1)
                correct += (preds == Y).sum().item()
                total += Y.numel()
        best_val = max(best_val, correct / total)

    return {"val_acc": best_val}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_probing(
    model_path: str = "checkpoints/chess_gpt.pt",
    dataset_path: str = "checkpoints/chess_dataset.pkl",
    vocab_path: str = "checkpoints/vocab.json",
    target_layer: int = -1,
    max_samples: int = 8_000,
    device: str = "cpu",
):
    vocab = load_vocab(vocab_path)
    vocab_size = len(vocab)

    model = ChessGPT(
        n_layers=8, d_model=128, n_heads=4,
        max_len=80, vocab_size=vocab_size + 1,
    ).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    print("ChessGPT loaded.")

    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    moves, boards = data["moves"], data["boards"]

    probe_moves  = moves[-2000:]
    probe_boards = boards[-2000:]

    print("Extracting activations...")
    extracted = extract_activations(
        model, probe_moves, probe_boards, vocab,
        max_samples=max_samples, device=device,
    )

    n = len(extracted["boards"])
    split = int(0.8 * n)

    layers = range(model.n_layers) if target_layer < 0 else [target_layer]

    print("\n" + "=" * 65)
    print(f"{'Layer':<8} {'Linear Probe':<18} {'MLP Probe':<18} {'Delta'}")
    print("=" * 65)

    results = {}
    for layer in layers:
        acts = extracted["activations"][layer]
        bds  = extracted["boards"]
        ta, va = acts[:split], acts[split:]
        tb, vb = bds[:split], bds[split:]

        lin = train_probe(LinearProbe(128), ta, tb, va, vb, epochs=20, device=device)
        mlp = train_probe(NonLinearProbe(128), ta, tb, va, vb, epochs=20, device=device)

        delta = mlp["val_acc"] - lin["val_acc"]
        print(f"  {layer:<6} {lin['val_acc']:.3f}             "
              f"{mlp['val_acc']:.3f}             "
              f"+{delta:.3f}")

        results[layer] = {"linear_acc": lin["val_acc"], "mlp_acc": mlp["val_acc"]}

    print("=" * 65)

    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/chess_probe_results.pkl", "wb") as f:
        pickle.dump(results, f)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="checkpoints/chess_gpt.pt")
    parser.add_argument("--dataset", default="checkpoints/chess_dataset.pkl")
    parser.add_argument("--vocab",   default="checkpoints/vocab.json")
    parser.add_argument("--layer",   type=int, default=-1)
    parser.add_argument("--max_samples", type=int, default=8_000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_probing(args.model, args.dataset, args.vocab,
                args.layer, args.max_samples, device)
