from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.block import TransformerBlock


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_embd: int
    num_heads: int
    num_layers: int
    activation: str = "relu"
    dropout: float = 0.1

class GPTLanguageModel(nn.Module):
    """
    Decoder-only GPT-style language model.
    
    Current version:
    token embedding + position embedding
    + stacked Transformer blocks
    + final LayerNorm + language-modeling head
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.n_embd
        )  # weight[vocab_size, n_embd]

        self.position_embedding = nn.Embedding(
            num_embeddings=config.block_size,
            embedding_dim=config.n_embd
        )  # weight[block_size, n_embd]

        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    n_embd=config.n_embd,
                    num_heads=config.num_heads,
                    block_size=config.block_size,
                    activation=config.activation,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_ln = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(
            in_features=config.n_embd,
            out_features=config.vocab_size,
        )  # weight[vocab_size, n_embd], bias[vocab_size]

    def forward(
            self,
            idx: torch.Tensor,
            targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            idx:token ids, shape [B, T]
            targets: next-token labels, shape [B, T]
            
        Returns:
            logits: shape [B, T, vocab_size]
            loss: cross entropy loss if targets is given
        """
        batch_size, seq_len = idx.shape

        if seq_len > self.config.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds block_size {self.config.block_size}."
            )
        
        token_emb = self.token_embedding(idx)  # [B, T, n_embd]

        positions = torch.arange(seq_len, device=idx.device)  # [T]
        pos_emb = self.position_embedding(positions)  # [T, n_embd]

        x = token_emb + pos_emb  # [B, T, n_embd]
        x = self.dropout(x)  # [B, T, n_embd]
        x = self.blocks(x)  # [B, T, n_embd]
        x = self.final_ln(x)  # [B, T, n_embd]

        logits = self.lm_head(x)  # [B, T, vocab_size]
        
        loss = None
        if targets is not None:
            batch_size, seq_len, vocab_size = logits.shape

            logits_flat = logits.view(batch_size * seq_len, vocab_size)  # [B * T, vocab_size]
            targets_flat = targets.view(batch_size * seq_len)  # [B * T]

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss