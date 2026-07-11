import torch
import torch.nn as nn

from src.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Input and output:
        x: [B, T, n_embd]
    """

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block.
    
    Input and output:
        x: [B, T, n_embd]
    """

    def __init__(
            self,
            n_embd: int,
            num_heads: int,
            block_size: int,
            dropout: float,            
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.attention = MultiHeadAttention(
            n_embd=n_embd,
            num_heads=num_heads,
            block_size=block_size,
            dropout=dropout,
        )

        self.ln2 = nn.LayerNorm(n_embd)
        self.feed_forward = FeedForward(
            n_embd=n_embd,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))

        return x