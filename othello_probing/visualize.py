"""
Visualizations for the Othello-GPT probing article.

Generates publication-ready figures:
    1. Probe accuracy per layer (linear vs MLP) — bar chart
    2. Board state reconstruction: ground truth vs decoded — side by side
    3. Heatmap of per-square probe accuracy

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

from model import OthelloGPT
from probe import (
    NonLinearProbe, extract_activations, ProbeDataset
)
from torch.utils.data import DataLoader


# Consistent styling for the article
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def plot_probe_accuracy_per_layer(results: dict, save_path: str = "figures/probe_accuracy.png"):
    """
    Bar chart: linear vs MLP probe accuracy at each layer.
    This is THE key figure of the article.
    """
    layers = sorted(results.keys())
    linear_accs = [results[l]["linear_acc"] for l in layers]
    mlp_accs = [results[l]["mlp_acc"] for l in layers]

    x = np.arange(len(layers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, linear_accs, width, label="Probe lineaire",
                   color="#6c757d", edgecolor="white")
    bars2 = ax.bar(x + width / 2, mlp_accs, width, label="Probe MLP (non-lineaire)",
                   color="#0d6efd", edgecolor="white")

    ax.set_xlabel("Couche du transformer")
    ax.set_ylabel("Precision (accuracy)")
    ax.set_title("Peut-on decoder la position du plateau\na partir des activations internes ?")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Couche {l}" for l in layers])
    ax.set_ylim(0.3, 1.05)
    ax.legend(loc="lower right")
    ax.axhline(y=1 / 3, color="red", linestyle="--", alpha=0.5, label="Hasard (33%)")
    ax.grid(axis="y", alpha=0.3)

    # Annotate best MLP accuracy
    best_layer = max(layers, key=lambda l: results[l]["mlp_acc"])
    best_acc = results[best_layer]["mlp_acc"]
    ax.annotate(
        f"{best_acc:.1%}",
        xy=(best_layer + width / 2, best_acc),
        xytext=(best_layer + 1.5, best_acc + 0.02),
        arrowprops=dict(arrowstyle="->", color="#0d6efd"),
        fontsize=12, fontweight="bold", color="#0d6efd",
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_board_comparison(
    ground_truth: np.ndarray,
    decoded: np.ndarray,
    move_idx: int = 0,
    save_path: str = "figures/board_comparison.png",
):
    """
    Side-by-side: ground truth board vs probe-decoded board.
    Shows that the MLP probe reconstructs the board almost perfectly.
    """
    cmap = ListedColormap(["#2d6a4f", "#1a1a1a", "#f5f5dc"])  # empty=green, black, white
    labels = ["Vide", "Noir", "Blanc"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, board, title in [
        (axes[0], ground_truth, "Plateau reel (verite terrain)"),
        (axes[1], decoded, "Plateau decode par la probe MLP"),
    ]:
        # Remap: -1->2, 0->0, 1->1 for colormap
        display = board.copy()
        display[board == -1] = 2

        ax.imshow(display, cmap=cmap, vmin=0, vmax=2)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(list("abcdefgh"))
        ax.set_yticklabels(range(1, 9))

        # Grid
        for i in range(9):
            ax.axhline(i - 0.5, color="black", linewidth=0.5)
            ax.axvline(i - 0.5, color="black", linewidth=0.5)

    # Legend
    patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip(["#2d6a4f", "#1a1a1a", "#f5f5dc"], labels)]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               frameon=True, fontsize=10)

    # Highlight differences
    diff = (ground_truth != decoded)
    n_errors = diff.sum()
    fig.suptitle(
        f"Position au coup #{move_idx} — "
        f"{'Reconstruction parfaite !' if n_errors == 0 else f'{n_errors} erreur(s)'}",
        fontsize=13, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_per_square_accuracy(
    probe: NonLinearProbe,
    val_acts: np.ndarray,
    val_boards: np.ndarray,
    save_path: str = "figures/per_square_accuracy.png",
    device: str = "cpu",
):
    """
    Heatmap 8x8: accuracy of the probe per square.
    Reveals which parts of the board are easier to decode.
    """
    probe.eval()
    ds = ProbeDataset(val_acts, val_boards)
    loader = DataLoader(ds, batch_size=512)

    per_square_correct = np.zeros(64)
    per_square_total = np.zeros(64)

    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            preds = probe(X).argmax(dim=-1)  # (B, 64)
            for sq in range(64):
                per_square_correct[sq] += (preds[:, sq] == Y[:, sq]).sum().item()
                per_square_total[sq] += len(Y)

    acc_map = (per_square_correct / per_square_total).reshape(8, 8)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(acc_map, cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_title("Precision de la probe par case", fontweight="bold")
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(list("abcdefgh"))
    ax.set_yticklabels(range(1, 9))

    # Annotate each cell
    for r in range(8):
        for c in range(8):
            val = acc_map[r, c]
            color = "white" if val < 0.75 else "black"
            ax.text(c, r, f"{val:.0%}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main: generate all figures
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load probe results
    results_path = "checkpoints/probe_results.pkl"
    if os.path.exists(results_path):
        with open(results_path, "rb") as f:
            results = pickle.load(f)
        print("Loaded probe results.")
        plot_probe_accuracy_per_layer(results)
    else:
        print(f"No probe results found at {results_path}. Run probe.py first.")

    # Load model + dataset for board comparison
    model_path = "checkpoints/othello_gpt.pt"
    dataset_path = "checkpoints/dataset.pkl"

    if os.path.exists(model_path) and os.path.exists(dataset_path):
        model = OthelloGPT(n_layers=8, d_model=128, n_heads=4).to(device)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        model.eval()

        with open(dataset_path, "rb") as f:
            data = pickle.load(f)
        moves, boards = data["moves"], data["boards"]

        # Pick a game from the end of the dataset (unseen during training)
        game_moves = moves[-10]
        game_boards = boards[-10]

        # Extract activations for this game
        extracted = extract_activations(
            model, [game_moves], [game_boards],
            max_samples=len(game_moves), device=device,
        )

        # Train a probe on the best layer (layer 6 typically)
        best_layer = 6
        acts = extracted["activations"][best_layer]
        bds = extracted["boards"]

        # Use a pre-trained probe if available, else train a quick one
        mlp_probe = NonLinearProbe(d_model=128, hidden=256).to(device)

        # Quick train on available data
        from probe import train_probe as _train_probe
        # We need more data for a good probe, extract from more games
        probe_moves = moves[-2000:]
        probe_boards = boards[-2000:]
        big_extracted = extract_activations(
            model, probe_moves, probe_boards,
            max_samples=8000, device=device,
        )
        big_acts = big_extracted["activations"][best_layer]
        big_bds = big_extracted["boards"]

        split = int(0.8 * len(big_bds))
        _train_probe(
            mlp_probe,
            big_acts[:split], big_bds[:split],
            big_acts[split:], big_bds[split:],
            epochs=20, device=device,
        )

        # Decode a specific position
        mlp_probe.eval()
        with torch.no_grad():
            test_act = torch.tensor(acts, dtype=torch.float32).to(device)
            decoded_all = mlp_probe(test_act).argmax(dim=-1).cpu().numpy()

        # Pick position at move 20 (mid-game, interesting)
        mid = min(20, len(game_moves) - 1)
        gt_board = game_boards[mid]
        decoded_board = decoded_all[mid].reshape(8, 8)
        # Remap back: 0->0, 1->1, 2->-1
        decoded_board_orig = decoded_board.copy().astype(np.int8)
        decoded_board_orig[decoded_board == 2] = -1

        plot_board_comparison(gt_board, decoded_board_orig, move_idx=mid)

        # Per-square accuracy heatmap
        plot_per_square_accuracy(
            mlp_probe,
            big_acts[split:], big_bds[split:],
            device=device,
        )

    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
