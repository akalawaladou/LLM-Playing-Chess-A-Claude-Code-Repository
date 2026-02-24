"""
Minimal GPT-style transformer for Othello move prediction.

Architecture mirrors the Othello-GPT paper (Li et al., ICLR 2023):
- Decoder-only transformer (causal attention)
- Input: sequence of move tokens (0..63)
- Output: next-move logits over 64 positions
- Key feature: we can extract intermediate activations for probing

The model is intentionally small (8 layers, 128-dim) so it trains
in minutes on CPU — perfect for an article demonstration.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Causal mask: positions can only attend to previous positions
        mask = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ChessGPT(nn.Module):
    """
    Small GPT trained to predict the next Chess move.

    Vocabulary: dynamic size (all UCI moves seen in training) + 1 padding token.
    """

    def __init__(
        self,
        n_layers: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        max_len: int = 80,      # max game length in Othello ~ 60
        vocab_size: int = 65,    # 64 squares + 1 pad token
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.n_layers = n_layers

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_len, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        return_activations: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            idx: (B, T) tensor of move token indices
            return_activations: if True, also return per-layer activations

        Returns:
            dict with keys:
                "logits": (B, T, vocab_size)
                "activations": dict mapping layer_idx -> (B, T, d_model)
                               (only if return_activations=True)
        """
        B, T = idx.shape
        assert T <= self.max_len, f"Sequence length {T} > max_len {self.max_len}"

        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        activations = {}

        for i, block in enumerate(self.blocks):
            x = block(x)
            if return_activations:
                activations[i] = x.detach().clone()

        x = self.ln_f(x)
        logits = self.head(x)

        result = {"logits": logits}
        if return_activations:
            result["activations"] = activations
        return result


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = ChessGPT(n_layers=8, d_model=128, n_heads=4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ChessGPT: {n_params:,} parameters")

    dummy = torch.randint(0, 64, (2, 20))
    out = model(dummy, return_activations=True)
    print(f"Logits shape: {out['logits'].shape}")
    print(f"Activations: {len(out['activations'])} layers, "
          f"each {out['activations'][0].shape}")
