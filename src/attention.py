import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Single-head causal self-attention.

    Input:
        x: [B, T, C]

    Output:
        out: [B, T, head_size]
    """

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.shape

        k = self.key(x)   # [B, T, head_size]
        q = self.query(x)  # [B, T, head_size]
        v = self.value(x)  # [B, T, head_size]

        scores = q @ k.transpose(-2, -1) # [B, T, T]
        scores = scores / math.sqrt(k.shape[-1])

        scores = scores.masked_fill(
            self.tril[:seq_len, :seq_len] == 0,
            float("-inf"),
        )

        weights = F.softmax(scores, dim=-1)  # [B, T, T]
        weights = self.dropout(weights)

        out = weights @ v  # [B, T, head_size]

        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple causal self-attention heads running in parallel

    Input:
        x: [B, T, n_embd]

    Output:
        out: [B, T, n_embd]
    """

    def __init__(
            self,
            n_embd: int,
            num_heads: int,
            block_size: int,
            dropout: float,
    ):

        super().__init__()

        assert n_embd % num_heads == 0, (
            "n_embd must be divisible by num_heads."
        )

        head_size = n_embd // num_heads

        self.heads = nn.ModuleList(
            [
                CausalSelfAttention(
                    n_embd=n_embd,
                    head_size=head_size,
                    block_size=block_size,
                    dropout=dropout
                )
                for _ in range(num_heads)
            ]
        )

        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_outputs = [head(x) for head in self.heads]

        out = torch.cat(head_outputs, dim=-1)
        out = self.proj(out)
        out = self.dropout(out)

        return out