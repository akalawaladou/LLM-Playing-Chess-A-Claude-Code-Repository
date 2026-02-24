"""
Probing analysis: can we decode the board state from OthelloGPT's activations?

This is the KEY experiment from the paper:
    - Extract activations from each transformer layer
    - Train a probe (linear or non-linear) to predict the board state
    - Compare: linear probe vs MLP probe

The probe's task:
    For each of the 64 squares, classify it as: empty (0), black (1), or white (2).
    Input:  activation vector at a given sequence position (d_model-dimensional)
    Output: 64 * 3 = 192 logits

If the probe succeeds, it means the model has *learned* to represent the
board state internally — even though it was only trained to predict moves.

Usage:
    python probe.py                  # run full probing analysis
    python probe.py --layer 5        # probe a specific layer only
"""

import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import OthelloGPT


# ---------------------------------------------------------------------------
# Probe architectures
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """
    Probe lineaire: une seule couche lineaire.
    Si ca marche -> l'info est lineairement separable dans les activations.
    Spoiler: ca marche moyennement (comme dans le papier).
    """
    def __init__(self, d_model: int, n_squares: int = 64, n_classes: int = 3):
        super().__init__()
        self.linear = nn.Linear(d_model, n_squares * n_classes)
        self.n_squares = n_squares
        self.n_classes = n_classes

    def forward(self, x):
        # x: (B, d_model)
        return self.linear(x).view(-1, self.n_squares, self.n_classes)


class NonLinearProbe(nn.Module):
    """
    Probe non-lineaire (MLP a 1 couche cachee).
    C'est celui qui reussit presque parfaitement dans le papier:
    l'info EST dans les activations, mais encodee de maniere non-lineaire.
    """
    def __init__(self, d_model: int, hidden: int = 256,
                 n_squares: int = 64, n_classes: int = 3):
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
    model: OthelloGPT,
    games_moves: list,
    games_boards: list,
    max_samples: int = 10_000,
    device: str = "cpu",
) -> dict:
    """
    Pass games through the model and collect:
        - activations at each layer for each (game, position)
        - ground-truth board state at each (game, position)

    Returns dict:
        "activations": {layer_idx: np.array (N, d_model)}
        "boards":      np.array (N, 8, 8)  values in {-1, 0, 1}
    """
    model.eval()
    layer_acts = {i: [] for i in range(model.n_layers)}
    board_labels = []
    n_collected = 0

    with torch.no_grad():
        for moves, boards in zip(games_moves, games_boards):
            if n_collected >= max_samples:
                break

            seq = torch.tensor(moves, dtype=torch.long).unsqueeze(0).to(device)
            out = model(seq, return_activations=True)

            # For each position in the game, collect activation + board state
            for t in range(len(moves)):
                if n_collected >= max_samples:
                    break
                for layer_idx, act_tensor in out["activations"].items():
                    # act_tensor: (1, T, d_model) -> take position t
                    layer_acts[layer_idx].append(
                        act_tensor[0, t].cpu().numpy()
                    )
                board_labels.append(boards[t].copy())
                n_collected += 1

    result = {
        "activations": {
            k: np.stack(v) for k, v in layer_acts.items()
        },
        "boards": np.stack(board_labels),
    }
    print(f"Extracted {n_collected} (position, activation) pairs")
    return result


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

class ProbeDataset(Dataset):
    def __init__(self, activations: np.ndarray, boards: np.ndarray):
        self.X = torch.tensor(activations, dtype=torch.float32)
        # Remap board values: -1 -> 2, 0 -> 0, 1 -> 1
        boards_remapped = boards.copy()
        boards_remapped[boards_remapped == -1] = 2
        self.Y = torch.tensor(
            boards_remapped.reshape(-1, 64), dtype=torch.long
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def train_probe(
    probe: nn.Module,
    train_acts: np.ndarray,
    train_boards: np.ndarray,
    val_acts: np.ndarray,
    val_boards: np.ndarray,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: str = "cpu",
) -> dict:
    """Train a probe and return train/val accuracies."""
    probe = probe.to(device)
    train_ds = ProbeDataset(train_acts, train_boards)
    val_ds = ProbeDataset(val_acts, val_boards)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        probe.train()
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            logits = probe(X)  # (B, 64, 3)
            loss = criterion(logits.reshape(-1, 3), Y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation accuracy
        probe.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                preds = probe(X).argmax(dim=-1)  # (B, 64)
                correct += (preds == Y).sum().item()
                total += Y.numel()

        val_acc = correct / total
        best_val_acc = max(best_val_acc, val_acc)

    return {"val_acc": best_val_acc}


# ---------------------------------------------------------------------------
# Main probing experiment
# ---------------------------------------------------------------------------

def run_probing(
    model_path: str = "checkpoints/othello_gpt.pt",
    dataset_path: str = "checkpoints/dataset.pkl",
    target_layer: int = -1,
    max_samples: int = 10_000,
    device: str = "cpu",
):
    # Load model
    model = OthelloGPT(n_layers=8, d_model=128, n_heads=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded.")

    # Load dataset
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    moves, boards = data["moves"], data["boards"]

    # Use last 2000 games for probing (avoid data seen during training)
    probe_moves = moves[-2000:]
    probe_boards = boards[-2000:]

    # Extract activations
    print("Extracting activations...")
    extracted = extract_activations(
        model, probe_moves, probe_boards,
        max_samples=max_samples, device=device,
    )

    # Train/val split for probing
    n = len(extracted["boards"])
    split = int(0.8 * n)

    layers_to_probe = (
        range(model.n_layers) if target_layer < 0
        else [target_layer]
    )

    print("\n" + "=" * 65)
    print(f"{'Layer':<8} {'Linear Probe':<18} {'MLP Probe':<18} {'Delta'}")
    print("=" * 65)

    results = {}

    for layer in layers_to_probe:
        acts = extracted["activations"][layer]
        bds = extracted["boards"]

        train_acts, val_acts = acts[:split], acts[split:]
        train_bds, val_bds = bds[:split], bds[split:]

        # --- Linear probe ---
        lin_probe = LinearProbe(d_model=128)
        lin_res = train_probe(
            lin_probe, train_acts, train_bds, val_acts, val_bds,
            epochs=20, device=device,
        )

        # --- Non-linear probe (MLP) ---
        mlp_probe = NonLinearProbe(d_model=128, hidden=256)
        mlp_res = train_probe(
            mlp_probe, train_acts, train_bds, val_acts, val_bds,
            epochs=20, device=device,
        )

        delta = mlp_res["val_acc"] - lin_res["val_acc"]
        print(f"  {layer:<6} {lin_res['val_acc']:.3f}             "
              f"{mlp_res['val_acc']:.3f}             "
              f"+{delta:.3f}")

        results[layer] = {
            "linear_acc": lin_res["val_acc"],
            "mlp_acc": mlp_res["val_acc"],
        }

    print("=" * 65)
    print("\nInterpretation:")
    print("  - Linear probe  ~ 60-75%: l'info existe mais n'est PAS")
    print("    lineairement separable dans les activations.")
    print("  - MLP probe     ~ 85-99%: un petit reseau non-lineaire")
    print("    reussit a decoder la position => le modele MAINTIENT")
    print("    un etat interne du plateau !")

    # Save results for visualization
    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/probe_results.pkl", "wb") as f:
        pickle.dump(results, f)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/othello_gpt.pt")
    parser.add_argument("--dataset", default="checkpoints/dataset.pkl")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Layer to probe (-1 = all layers)")
    parser.add_argument("--max_samples", type=int, default=10_000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_probing(
        model_path=args.model,
        dataset_path=args.dataset,
        target_layer=args.layer,
        max_samples=args.max_samples,
        device=device,
    )
