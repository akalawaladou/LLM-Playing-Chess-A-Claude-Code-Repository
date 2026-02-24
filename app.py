"""
Streamlit interactive demo — LLM Board Game Probing

Visualise how OthelloGPT and ChessGPT encode the board state
in their internal activations (neuron activations).

Usage:
    streamlit run app.py
"""

import os
import sys
import pickle
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(__file__)


# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Board Game Probing",
    page_icon="♟",
    layout="wide",
)

st.title("♟ LLM Board Game Probing")
st.markdown(
    """
    **Que cache un LLM entraîné sur des parties de jeu ?**
    Ce démo visualise comment un petit GPT apprend à maintenir en interne
    une *représentation du plateau* — même s'il n'a jamais « vu » un échiquier.
    Sélectionnez un jeu, une partie, un coup et une couche, puis explorez les
    trois onglets.
    """
)

# ── sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Paramètres")

    game_choice = st.radio("Jeu", ["Othello", "Échecs"], horizontal=True)
    game_key = "othello" if game_choice == "Othello" else "chess"

    layer = st.slider("Couche du transformer", 0, 7, 6)

    st.markdown("---")
    st.caption(
        "**Probe linéaire** : régression logistique sur les activations.  \n"
        "**Probe MLP** : petit réseau non-linéaire. Si le MLP >> linéaire, "
        "l'info est encodée de façon non-linéaire dans le modèle."
    )


# ── helpers: load models & data ──────────────────────────────────────────────

def ckpt(game, filename):
    base = "othello_probing" if game == "othello" else "chess_probing"
    return os.path.join(ROOT, base, "checkpoints", filename)


@st.cache_resource(show_spinner="Chargement du modèle…")
def load_othello_model():
    from othello_probing.model import OthelloGPT  # type: ignore
    path = ckpt("othello", "othello_gpt.pt")
    if not os.path.exists(path):
        return None
    model = OthelloGPT(n_layers=8, d_model=128, n_heads=4)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


@st.cache_resource(show_spinner="Chargement du modèle…")
def load_chess_model():
    from chess_probing.model import ChessGPT  # type: ignore
    from chess_probing.chess_engine import load_vocab  # type: ignore
    vocab_path = ckpt("chess", "vocab.json")
    model_path = ckpt("chess", "chess_gpt.pt")
    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        return None, None
    vocab = load_vocab(vocab_path)
    model = ChessGPT(n_layers=8, d_model=128, n_heads=4,
                     max_len=80, vocab_size=len(vocab) + 1)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model, vocab


@st.cache_data(show_spinner="Chargement du dataset…")
def load_dataset(game):
    fname = "dataset.pkl" if game == "othello" else "chess_dataset.pkl"
    path = ckpt(game, fname)
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["moves"], data["boards"]


@st.cache_data(show_spinner="Extraction des activations…")
def get_activations_othello(game_idx, _model):
    moves_list, boards_list = load_dataset("othello")
    moves  = moves_list[-10:][game_idx]
    boards = boards_list[-10:][game_idx]

    seq = torch.tensor(moves, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        out = _model(seq, return_activations=True)

    # layer -> (T, 128)
    acts = {k: v[0].cpu().numpy() for k, v in out["activations"].items()}
    return acts, boards, moves


@st.cache_data(show_spinner="Extraction des activations…")
def get_activations_chess(game_idx, _model, vocab):
    moves_list, boards_list = load_dataset("chess")
    moves  = moves_list[-10:][game_idx]
    boards = boards_list[-10:][game_idx]

    tokens = [vocab[m] for m in moves if m in vocab]
    tokens = tokens[:79]
    seq = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        out = _model(seq, return_activations=True)

    acts = {k: v[0].cpu().numpy() for k, v in out["activations"].items()}
    return acts, boards, moves


@st.cache_resource(show_spinner="Entraînement de la probe…")
def train_mlp_probe(game, _model, vocab=None):
    """Train a small MLP probe on the last 1000 positions of held-out games."""
    from othello_probing.probe import (  # type: ignore
        NonLinearProbe as OthelloMLP, extract_activations as othello_extract,
        train_probe as othello_train,
    )
    from chess_probing.probe import (    # type: ignore
        NonLinearProbe as ChessMLP, extract_activations as chess_extract,
        train_probe as chess_train, N_CLASSES,
    )

    moves_list, boards_list = load_dataset(game)
    probe_moves  = moves_list[-1000:]
    probe_boards = boards_list[-1000:]

    if game == "othello":
        extracted = othello_extract(_model, probe_moves, probe_boards,
                                    max_samples=4000)
        acts = extracted["activations"][6]
        bds  = extracted["boards"]
        split = int(0.8 * len(bds))
        probe = OthelloMLP(d_model=128, hidden=256)
        othello_train(probe, acts[:split], bds[:split],
                      acts[split:], bds[split:], epochs=15)
        return probe, extracted
    else:
        extracted = chess_extract(_model, probe_moves, probe_boards, vocab,
                                  max_samples=4000)
        acts = extracted["activations"][6]
        bds  = extracted["boards"]
        split = int(0.8 * len(bds))
        probe = ChessMLP(d_model=128, hidden=256, n_classes=N_CLASSES)
        chess_train(probe, acts[:split], bds[:split],
                    acts[split:], bds[split:], epochs=15)
        return probe, extracted


# ── board drawing helpers ────────────────────────────────────────────────────

OTHELLO_COLORS = {0: "#2d6a4f", 1: "#1a1a1a", -1: "#f5f5dc"}
CHESS_PIECES = {
    0: "", 1: "♙", 2: "♘", 3: "♗", 4: "♖", 5: "♕", 6: "♔",
    7: "♟", 8: "♞", 9: "♝", 10: "♜", 11: "♛", 12: "♚",
}


def draw_othello_board(ax, board: np.ndarray, title: str):
    cmap_vals = np.vectorize(lambda v: {0: 0, 1: 1, -1: 2}[int(v)])(board)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#2d6a4f", "#1a1a1a", "#f5f5dc"])
    ax.imshow(cmap_vals, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    for i in range(9):
        ax.axhline(i - 0.5, color="black", lw=0.5)
        ax.axvline(i - 0.5, color="black", lw=0.5)
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels(list("abcdefgh"))
    ax.set_yticklabels(range(1, 9))
    ax.set_title(title, fontweight="bold", fontsize=11)


def draw_chess_board(ax, board: np.ndarray, title: str):
    for r in range(8):
        for c in range(8):
            color = "#f0d9b5" if (r + c) % 2 == 0 else "#b58863"
            ax.add_patch(plt.Rectangle((c, 7 - r), 1, 1, color=color))
            p = int(board[r, c])
            if p != 0:
                pc = "black" if 1 <= p <= 6 else "#1a1a6e"
                ax.text(c + 0.5, 7 - r + 0.5, CHESS_PIECES[p],
                        ha="center", va="center", fontsize=18,
                        color=pc, fontweight="bold")
    ax.set_xlim(0, 8); ax.set_ylim(0, 8)
    ax.set_xticks(np.arange(8) + 0.5); ax.set_yticks(np.arange(8) + 0.5)
    ax.set_xticklabels(list("abcdefgh"))
    ax.set_yticklabels(list("12345678"))
    ax.set_title(title, fontweight="bold", fontsize=11)
    for s in ax.spines.values(): s.set_linewidth(2)


def decode_board_from_activation(probe, act_vec, n_classes):
    probe.eval()
    with torch.no_grad():
        x = torch.tensor(act_vec, dtype=torch.float32).unsqueeze(0)
        pred = probe(x).argmax(dim=-1)[0].numpy().reshape(8, 8)
    if n_classes == 3:  # Othello: remap 2 -> -1
        pred = pred.astype(np.int8)
        pred[pred == 2] = -1
    return pred.astype(np.int8)


# ── main UI ──────────────────────────────────────────────────────────────────

def missing_ckpt_warning(game):
    base = "othello_probing" if game == "othello" else "chess_probing"
    st.warning(
        f"Aucun modèle entraîné trouvé pour **{game_choice}**. "
        f"Lance d'abord le script d'entraînement :\n\n"
        f"```bash\ncd {base}\n"
        f"python train.py --n_games 10000 --epochs 15\n"
        f"python probe.py\n```"
    )


# === Othello ===
if game_key == "othello":
    model = load_othello_model()
    if model is None:
        missing_ckpt_warning("othello")
        st.stop()

    moves_all, boards_all = load_dataset("othello")
    if moves_all is None:
        missing_ckpt_warning("othello")
        st.stop()

    sample_games = moves_all[-10:]
    game_idx = st.sidebar.selectbox(
        "Partie (parmi les 10 dernières)",
        range(min(10, len(sample_games))),
        format_func=lambda i: f"Partie {i+1} ({len(sample_games[i])} coups)",
    )

    acts, boards, moves = get_activations_othello(game_idx, model)
    move_idx = st.sidebar.slider("Coup n°", 0, len(moves) - 1,
                                  min(20, len(moves) - 1))

    probe, extracted = train_mlp_probe("othello", model)
    n_classes = 3

# === Chess ===
else:
    model, vocab = load_chess_model()
    if model is None:
        missing_ckpt_warning("chess")
        st.stop()

    moves_all, boards_all = load_dataset("chess")
    if moves_all is None:
        missing_ckpt_warning("chess")
        st.stop()

    sample_games = moves_all[-10:]
    game_idx = st.sidebar.selectbox(
        "Partie (parmi les 10 dernières)",
        range(min(10, len(sample_games))),
        format_func=lambda i: f"Partie {i+1} ({len(sample_games[i])} coups)",
    )

    acts, boards, moves = get_activations_chess(game_idx, model, vocab)
    move_idx = st.sidebar.slider("Coup n°", 0, len(moves) - 1,
                                  min(20, len(moves) - 1))

    probe, extracted = train_mlp_probe("chess", model, vocab)
    n_classes = 13


# ── tabs ─────────────────────────────────────────────────────────────────────
tab_boards, tab_probing, tab_heatmap = st.tabs(
    ["🗺️ Plateaux", "📊 Précision par couche", "🌡️ Heatmap par case"]
)

# ── Tab 1: Board comparison ───────────────────────────────────────────────────
with tab_boards:
    st.subheader(f"Coup #{move_idx} — plateau réel vs décodé par la probe MLP")

    act_vec = acts[layer][move_idx]
    gt_board = boards[move_idx]
    decoded  = decode_board_from_activation(probe, act_vec, n_classes)

    n_err = int((gt_board != decoded).sum())

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(4, 4))
        if game_key == "othello":
            draw_othello_board(ax, gt_board, "Plateau réel")
        else:
            draw_chess_board(ax, gt_board, "Plateau réel")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(4, 4))
        if game_key == "othello":
            draw_othello_board(ax, decoded, "Plateau décodé (probe MLP)")
        else:
            draw_chess_board(ax, decoded, "Plateau décodé (probe MLP)")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    if n_err == 0:
        st.success("Reconstruction parfaite : 0 erreur sur 64 cases !")
    else:
        pct = (64 - n_err) / 64
        st.info(f"{64-n_err}/64 cases correctes ({pct:.0%}) — couche {layer}")

    st.markdown("---")
    st.markdown(
        f"**Coup joué :** `{moves[move_idx]}` &nbsp;|&nbsp; "
        f"**Couche analysée :** {layer} &nbsp;|&nbsp; "
        f"**Erreurs :** {n_err}"
    )


# ── Tab 2: Probe accuracy per layer ──────────────────────────────────────────
with tab_probing:
    st.subheader("Précision des probes linéaire vs MLP par couche")

    # Load pre-computed results if available, else compute on the fly
    results_file = ckpt(game_key, f"{'chess_' if game_key=='chess' else ''}probe_results.pkl")

    if os.path.exists(results_file):
        with open(results_file, "rb") as f:
            results = pickle.load(f)
    else:
        st.info("Résultats de probing non trouvés — calcul en cours (quelques minutes)…")
        results = None

    if results:
        layers_sorted = sorted(results.keys())
        lin_vals = [results[l]["linear_acc"] for l in layers_sorted]
        mlp_vals = [results[l]["mlp_acc"]    for l in layers_sorted]

        x, w = np.arange(len(layers_sorted)), 0.35
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x - w/2, lin_vals, w, label="Probe linéaire",
               color="#6c757d", edgecolor="white")
        ax.bar(x + w/2, mlp_vals, w, label="Probe MLP",
               color="#0d6efd", edgecolor="white")
        baseline = 1/3 if game_key == "othello" else 1/13
        ax.axhline(baseline, color="red", linestyle="--", alpha=0.6,
                   label=f"Hasard ({baseline:.0%})")
        ax.axvline(layer - 0.5 + 0.5, color="orange", linestyle=":",
                   linewidth=2, label=f"Couche sélectionnée ({layer})")
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in layers_sorted])
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(
            f"{'OthelloGPT' if game_key=='othello' else 'ChessGPT'} — "
            f"Peut-on décoder la position depuis les activations ?"
        )
        st.pyplot(fig)
        plt.close()

        col1, col2, col3 = st.columns(3)
        col1.metric("Probe linéaire (couche sélectionnée)",
                    f"{results[layer]['linear_acc']:.1%}")
        col2.metric("Probe MLP (couche sélectionnée)",
                    f"{results[layer]['mlp_acc']:.1%}")
        col3.metric("Delta (MLP − linéaire)",
                    f"+{results[layer]['mlp_acc'] - results[layer]['linear_acc']:.1%}")
    else:
        st.markdown(
            "Lance `python probe.py` dans le dossier correspondant "
            "pour pré-calculer les résultats."
        )

    with st.expander("💡 Comment lire ce graphique ?"):
        st.markdown(
            """
            - **Probe linéaire** : une régression logistique branchée sur les
              activations d'une couche. Si elle réussit, l'info est *linéairement
              accessible*.
            - **Probe MLP** : un petit réseau à une couche cachée. Elle réussit
              beaucoup mieux → l'information est bien présente, mais encodée de
              façon **non-linéaire** dans l'espace des activations.
            - **Baseline (hasard)** : précision d'un classificateur aléatoire.
              Othello = 33 % (3 classes), Échecs = 7,7 % (13 classes).
            - **Pic aux couches intermédiaires** : les premières couches encodent
              surtout la syntaxe des coups ; les couches du milieu développent la
              représentation spatiale du plateau.
            """
        )


# ── Tab 3: Per-square heatmap ────────────────────────────────────────────────
with tab_heatmap:
    st.subheader("Précision de la probe par case (couche sélectionnée)")

    val_acts   = extracted["activations"][layer]
    val_boards = extracted["boards"]
    split = int(0.8 * len(val_boards))

    probe.eval()
    from torch.utils.data import DataLoader as DL
    if game_key == "othello":
        from othello_probing.probe import ProbeDataset  # type: ignore
    else:
        from chess_probing.probe import ProbeDataset    # type: ignore

    ds = ProbeDataset(val_acts[split:], val_boards[split:])
    loader = DL(ds, batch_size=512)
    correct = np.zeros(64)
    total   = np.zeros(64)

    with torch.no_grad():
        for X, Y in loader:
            preds = probe(X).argmax(dim=-1)
            for sq in range(64):
                correct[sq] += (preds[:, sq] == Y[:, sq]).sum().item()
                total[sq]   += len(Y)

    acc_map = (correct / np.maximum(total, 1)).reshape(8, 8)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(acc_map, cmap="RdYlGn", vmin=0.3, vmax=1.0,
                   interpolation="nearest")
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels(list("abcdefgh"))
    ax.set_yticklabels(range(1, 9))
    ax.set_title(f"Précision par case — couche {layer}", fontweight="bold")

    for r in range(8):
        for c in range(8):
            val = acc_map[r, c]
            color = "white" if val < 0.55 else "black"
            ax.text(c, r, f"{val:.0%}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)
    st.pyplot(fig)
    plt.close()

    with st.expander("💡 Comment lire cette heatmap ?"):
        st.markdown(
            """
            Chaque case montre avec quelle précision la probe **MLP** peut deviner
            l'état de cette case à partir du vecteur d'activation à la couche
            sélectionnée.

            - **Cases vertes (>80 %)** : bien représentées dans les activations.
            - **Cases rouges (<50 %)** : moins bien trackées par le modèle.

            **Pourquoi le centre est-il souvent plus difficile ?**
            Aux échecs comme à Othello, les cases centrales changent d'état très
            souvent (pièces mobiles, retournements). Le modèle doit donc mettre
            à jour sa représentation interne plus fréquemment, ce qui est plus
            difficile à apprendre avec peu de données.

            **Ce que montre cette carte** : le modèle *maintient bien* un état
            interne du plateau, même si la représentation est imparfaite pour les
            cases les plus dynamiques.
            """
        )

# ── footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Basé sur : Li et al., *Emergent World Representations: Exploring a Sequence "
    "Model Trained on a Synthetic Task* (ICLR 2023). "
    "Code : [GitHub](https://github.com/akalawaladou/LLM-Playing-Chess-A-Claude-Code-Repository)"
)
