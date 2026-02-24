# LLM Board Game Probing — Othello & Chess

> *Do language models secretly "see" the board? We train a tiny GPT on game moves and read its mind.*

## What is this?

This repo reproduces and extends the **Othello-GPT** experiment (Li et al., ICLR 2023):
we train a small transformer to predict the next move in Othello and Chess — **without
ever showing it a board** — and then probe its internal activations to see whether it
has learned an implicit representation of the board state.

A companion Medium article explains the results in detail (French):
[Les LLM hallucinent aux échecs et c'est normal, vraiment?](https://medium.com/@meryrami/les-llm-hallucinent-aux-%C3%A9checs-et-cest-normal-vraiment-080170a3d62a)

---

## The Othello-GPT Paper

**"Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task"**
Li et al., ICLR 2023

Key finding: a GPT trained *only* on sequences of Othello moves (no images, no board,
no rules explicitly given) spontaneously develops an internal representation of the
board state. This representation is:

1. **Present** — a small classifier can decode the board from internal activations.
2. **Causal** — modifying the activations (activation patching) changes the model's
   predicted moves as if the position had actually changed.
3. **Non-linear** — a linear classifier achieves ~65–70 %; a small MLP achieves ~95 %+.

We reproduce this experiment and extend it to **Chess**.

---

## What is Probing?

Probing is an interpretability technique that asks: *is a given piece of information
encoded in a model's internal activations?*

**Protocol:**
1. **Collect activations** — pass game sequences through the GPT; record the hidden
   vector at each transformer layer for each position in the game.
2. **Build labels** — for each position, record the ground-truth board state (which
   square holds which piece).
3. **Train a probe** — a small classifier that takes the activation vector as input
   and tries to predict the board state.

Two probes are compared:

| Probe | Architecture | What success means |
|---|---|---|
| **Linear** | `d_model → 64 × n_classes` | Info is linearly accessible |
| **MLP** | `d_model → 256 → 64 × n_classes` | Info is present but non-linearly encoded |

If the MLP probe greatly outperforms the linear probe, the board state is encoded in the
activations but in a *folded, non-linear* way — the model has developed its own internal
"coordinate system" for the board.

---

## Reading the Heatmap

The **per-square accuracy heatmap** (8×8 grid) shows how precisely the probe can decode
the state of each individual square.

```
High accuracy (green) = well represented in the model's activations
Low  accuracy (red)   = harder to decode
```

**Why is the center often harder?**

- **Othello**: central squares get flipped frequently; the model must track many updates.
- **Chess**: central squares host the most active pieces; high turnover = harder to track.
- **Edges & corners** start either empty (chess) or occupied (Othello standard start)
  and change state less often → easier for the model to represent.

This is not a bug — it reflects what is genuinely *harder* to track in each game.

---

## Our Results

*Trained on 10K random games, 10–15 epochs on CPU.*

| Metric | Othello | Chess |
|---|---|---|
| Random baseline | 33.3% (3 classes) | 7.7% (13 classes) |
| Linear probe (best layer) | ~69% | ~65% |
| MLP probe (best layer) | ~70%* | ~67%* |

*\* Models are undertrained (CPU demo). With 50K+ games and 30+ epochs the gap
widens to **linear ~65 % vs MLP ~95 %+**, matching the original paper.*

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the interactive demo

The demo requires pre-trained models. Train them first (see below), then:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`. The sidebar lets you:
- Switch between Othello and Chess
- Choose a sample game and move number
- Select which transformer layer to inspect

Three tabs:
- **Plateaux** — ground-truth board vs probe-decoded board side by side
- **Précision par couche** — bar chart of linear vs MLP probe accuracy per layer
- **Heatmap** — per-square accuracy for the selected layer

---

## Reproduce the Experiments

### Othello

```bash
cd othello_probing

# Train (10K games, 15 epochs, ~5 min on CPU)
python train.py --n_games 10000 --epochs 15

# Run probing analysis
python probe.py --max_samples 5000

# Generate figures
python visualize.py
```

### Chess

```bash
cd chess_probing

# Train (10K games, 15 epochs, ~15 min on CPU)
python train.py --n_games 10000 --epochs 15

# Run probing analysis
python probe.py --max_samples 5000

# Generate figures
python visualize.py
```

For better results (closer to the paper):

```bash
python train.py --n_games 50000 --epochs 30   # ~1h on GPU
python probe.py --max_samples 20000
```

---

## Repo Structure

```
.
├── app.py                          Streamlit interactive frontend
├── requirements.txt                All dependencies
├── article_continuation.md         Medium article (French)
│
├── othello_probing/
│   ├── othello.py                  Othello game engine (8x8, 3-class board)
│   ├── model.py                    OthelloGPT: 8-layer decoder transformer
│   ├── train.py                    Training pipeline (next-move prediction)
│   ├── probe.py                    Linear & MLP probes (3 classes/square)
│   ├── visualize.py                Publication-ready figures
│   ├── requirements.txt
│   ├── checkpoints/                Saved model + dataset (gitignored)
│   └── figures/                    Generated PNG figures
│
└── chess_probing/
    ├── chess_engine.py             Chess game engine (pure numpy, 13-class board)
    ├── model.py                    ChessGPT: same architecture, larger vocab
    ├── train.py                    Training pipeline (UCI move tokens)
    ├── probe.py                    Linear & MLP probes (13 classes/square)
    ├── visualize.py                Chess-specific figures
    ├── requirements.txt
    ├── checkpoints/                Saved model + dataset + vocab (gitignored)
    └── figures/                    Generated PNG figures
```

### Board encodings

**Othello** (3 classes):

| Value | Meaning |
|---|---|
| `0` | Empty |
| `1` | Black piece |
| `-1` | White piece |

**Chess** (13 classes):

| Value | Piece |
|---|---|
| `0` | Empty |
| `1-6` | White Pawn, Knight, Bishop, Rook, Queen, King |
| `7-12` | Black Pawn, Knight, Bishop, Rook, Queen, King |

### Model architecture

Both models use the same **decoder-only transformer**:

| Parameter | Value |
|---|---|
| Layers | 8 |
| Hidden dim | 128 |
| Attention heads | 4 |
| Parameters | ~1.6M |
| Training objective | Next-token cross-entropy |

---

## References

- Li et al., [Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task](https://arxiv.org/abs/2210.13382), ICLR 2023
- Nanda et al., [Actually, Othello-GPT Has A Linear Emergent World Representation](https://arxiv.org/abs/2310.07582), 2023
- Alain & Bengio, [Understanding intermediate layers using linear classifier probes](https://arxiv.org/abs/1610.01644), 2016
