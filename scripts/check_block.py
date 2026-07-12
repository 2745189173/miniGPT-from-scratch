import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.block import FeedForward, TransformerBlock


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 2
    seq_len = 5
    n_embd = 16
    dropout = 0.0
    block_size = 8
    num_heads = 4

    feed_forward = FeedForward(
        n_embd=n_embd,
        dropout=dropout,
        activation="gelu",
    ).to(device)
    assert any(
        isinstance(module, nn.GELU)
        for module in feed_forward.modules()
    ), "FeedForward should use GELU activation."

    x = torch.randn(batch_size, seq_len, n_embd, device=device)
    out = feed_forward(x)

    print("device:", device)
    print("input shape:", x.shape)
    print("output shape:", out.shape)

    assert out.shape == x.shape, (
        "Feed-forward network must preserve [B, T, n_embd]."
    )

    assert torch.isfinite(out).all(), (
        "Feed-forward output should contain only finite values."
    )

    transformer_block = TransformerBlock(
        n_embd=n_embd,
        num_heads=num_heads,
        block_size=block_size,
        dropout=dropout,
        activation="gelu",
    ).to(device)

    block_out = transformer_block(x)

    print("Transformer block output shape:", block_out.shape)

    assert block_out.shape == x.shape, (
        "Transformer block must preserve [B, T, n_embd]."
    )

    assert torch.isfinite(block_out).all(), (
        "Transformer block output should contain only finite values."
    )

    print("\ncheck passed: feed-forward network and Transformer block are correct.")


if __name__ == "__main__":
    main()