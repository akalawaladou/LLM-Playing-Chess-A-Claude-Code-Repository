"""
Visualizations for ChessGPT probing.

Three figures (parallel to othello_probing/visualize.py):
    1. Probe accuracy per layer (linear vs MLP)
    2. Board comparison: ground truth vs MLP-decoded
    3. Per-square probe accuracy heatmap

Usage:
    python visualize.py
"""

import os
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

from chess_engine import load_vocab, PIECE_SYMBOLS
from model import ChessGPT
from probe import NonLinearProbe, extract_activations, ProbeDataset, N_CLASSES, train_probe

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#f8f8f8",
    "font.size": 11, "axes.titlesize": 13,
})

# Chess piece Unicode symbols (indexed 0..12)
UNICODE_PIECES = {
    0: "",
    1: "♙", 2: "♘", 3: "♗", 4: "♖", 5: "♕", 6: "♔",
    7: "♟", 8: "♞", 9: "♝", 10: "♜", 11: "♛", 12: "♚",
}


def plot_probe_accuracy_per_layer(
    results: dict,
    save_path: str = "figures/chess_probe_accuracy.png",
):
    layers = sorted(results.keys())
    lin = [results[l]["linear_acc"] for l in layers]
    mlp = [results[l]["mlp_acc"]    for l in layers]

    x, w = np.arange(len(layers)), 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, lin, w, label="Probe linéaire", color="#6c757d", edgecolor="white")
    ax.bar(x + w/2, mlp, w, label="Probe MLP", color="#0d6efd", edgecolor="white")
    ax.axhline(1/13, color="red", linestyle="--", alpha=0.5,
               label=f"Hasard ({1/13:.0%})")
    ax.set_xlabel("Couche du transformer")
    ax.set_ylabel("Précision (accuracy)")
    ax.set_title("ChessGPT — Décodage de la position\nà partir des activations internes")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Couche {l}" for l in layers])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    best = max(layers, key=lambda l: results[l]["mlp_acc"])
    best_acc = results[best]["mlp_acc"]
    ax.annotate(f"{best_acc:.1%}", xy=(best + w/2, best_acc),
                xytext=(best + 1.5, best_acc + 0.03),
                arrowprops=dict(arrowstyle="->", color="#0d6efd"),
                fontsize=12, fontweight="bold", color="#0d6efd")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def _draw_chess_board(ax, board: np.ndarray, title: str):
    """Draw an 8x8 chess board with piece symbols on matplotlib Axes."""
    light = "#f0d9b5"
    dark  = "#b58863"
    for r in range(8):
        for c in range(8):
            color = light if (r + c) % 2 == 0 else dark
            ax.add_patch(plt.Rectangle((c, 7 - r), 1, 1, color=color))
            p = int(board[r, c])
            if p != 0:
                symbol = UNICODE_PIECES[p]
                piece_color = "black" if 1 <= p <= 6 else "#1a1a6e"
                ax.text(c + 0.5, 7 - r + 0.5, symbol,
                        ha="center", va="center",
                        fontsize=20, color=piece_color,
                        fontweight="bold",
                        fontfamily="DejaVu Sans")
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xticks(np.arange(8) + 0.5)
    ax.set_xticklabels(list("abcdefgh"))
    ax.set_yticks(np.arange(8) + 0.5)
    ax.set_yticklabels(list("12345678"))
    ax.set_title(title, fontweight="bold")
    # Border
    for spine in ax.spines.values():
        spine.set_linewidth(2)


def plot_board_comparison(
    ground_truth: np.ndarray,
    decoded: np.ndarray,
    move_idx: int = 0,
    save_path: str = "figures/chess_board_comparison.png",
):
    n_errors = (ground_truth != decoded).sum()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _draw_chess_board(axes[0], ground_truth, "Plateau réel (vérité terrain)")
    _draw_chess_board(axes[1], decoded,       "Plateau décodé par la probe MLP")

    msg = "Reconstruction parfaite !" if n_errors == 0 else f"{n_errors} erreur(s)"
    fig.suptitle(f"Position au coup #{move_idx} — {msg}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_per_square_accuracy(
    probe: NonLinearProbe,
    val_acts: np.ndarray,
    val_boards: np.ndarray,
    save_path: str = "figures/chess_per_square_accuracy.png",
    device: str = "cpu",
):
    probe.eval()
    loader = DataLoader(ProbeDataset(val_acts, val_boards), batch_size=512)
    correct = np.zeros(64)
    total   = np.zeros(64)

    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            preds = probe(X).argmax(dim=-1)
            for sq in range(64):
                correct[sq] += (preds[:, sq] == Y[:, sq]).sum().item()
                total[sq]   += len(Y)

    acc_map = (correct / np.maximum(total, 1)).reshape(8, 8)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(acc_map, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_title("Précision de la probe par case\n(ChessGPT)", fontweight="bold")
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(list("abcdefgh"))
    ax.set_yticklabels(range(1, 9))

    for r in range(8):
        for c in range(8):
            val = acc_map[r, c]
            color = "white" if val < 0.5 else "black"
            ax.text(c, r, f"{val:.0%}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Probe accuracy per layer
    results_path = "checkpoints/chess_probe_results.pkl"
    if os.path.exists(results_path):
        with open(results_path, "rb") as f:
            results = pickle.load(f)
        plot_probe_accuracy_per_layer(results)
    else:
        print(f"No probe results at {results_path} — run probe.py first.")

    # 2 & 3. Board comparison + per-square heatmap
    model_path   = "checkpoints/chess_gpt.pt"
    dataset_path = "checkpoints/chess_dataset.pkl"
    vocab_path   = "checkpoints/vocab.json"

    if not (os.path.exists(model_path) and os.path.exists(dataset_path)):
        print("Model or dataset missing — run train.py first.")
        return

    vocab = load_vocab(vocab_path)
    model = ChessGPT(
        n_layers=8, d_model=128, n_heads=4,
        max_len=80, vocab_size=len(vocab) + 1,
    ).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    moves, boards = data["moves"], data["boards"]

    # Extract for probing
    big_moves  = moves[-1500:]
    big_boards = boards[-1500:]
    big_ex = extract_activations(
        model, big_moves, big_boards, vocab,
        max_samples=6000, device=device,
    )

    best_layer = 6
    acts = big_ex["activations"][best_layer]
    bds  = big_ex["boards"]
    split = int(0.8 * len(bds))

    mlp = NonLinearProbe(128, n_classes=N_CLASSES).to(device)
    train_probe(mlp, acts[:split], bds[:split], acts[split:], bds[split:],
                epochs=20, device=device)

    # Single game for board comparison
    game_moves  = moves[-5]
    game_boards = boards[-5]
    game_ex = extract_activations(
        model, [game_moves], [game_boards], vocab,
        max_samples=len(game_moves), device=device,
    )
    game_acts = game_ex["activations"][best_layer]

    mlp.eval()
    with torch.no_grad():
        t_acts = torch.tensor(game_acts, dtype=torch.float32).to(device)
        decoded_all = mlp(t_acts).argmax(dim=-1).cpu().numpy()

    mid = min(20, len(game_moves) - 1)
    gt_board = game_boards[mid]
    decoded_board = decoded_all[mid].reshape(8, 8).astype(np.int8)

    plot_board_comparison(gt_board, decoded_board, move_idx=mid)
    plot_per_square_accuracy(mlp, acts[split:], bds[split:], device=device)

    print("\nAll chess figures saved to figures/")


if __name__ == "__main__":
    main()
