import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.attention import CausalSelfAttention, MultiHeadAttention


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 2
    block_size = 8
    seq_len = 5
    n_embd = 16
    head_size = 4
    dropout = 0.0

    attention = CausalSelfAttention(
        n_embd=n_embd,
        head_size=head_size,
        block_size=block_size,
        dropout=dropout,
    ).to(device)

    x = torch.randn(batch_size, seq_len, n_embd, device=device)
    out = attention(x)

    print("device:", device)
    print("x shape:", x.shape)
    print("out shape:", out.shape)
    print("causal mask:")
    print(attention.tril[:seq_len, :seq_len])

    assert out.shape == (
        batch_size,
        seq_len,
        head_size,
    ), "Attention output should have shape [batch_size, seq_len, head_size]."

    mask = attention.tril[:seq_len, :seq_len]
    assert torch.equal(mask, torch.tril(mask)), (
        "Causal mask should be lower triangular."
    )

    num_heads = 4

    multi_head_attention = MultiHeadAttention(
        n_embd=n_embd,
        num_heads=num_heads,
        block_size=block_size,
        dropout=dropout,
    ).to(device)

    multi_head_out = multi_head_attention(x)
    print("multi-head output shape:", multi_head_out.shape)

    assert multi_head_out.shape == (
        batch_size,
        seq_len,
        n_embd,
    ), "Multi-head attention output should have shape [batch_size, seq_len, n_embd]."

    print("\ncheck passed: single-head and multi-head causal attention are correct.")


if __name__ == "__main__":
    main()
