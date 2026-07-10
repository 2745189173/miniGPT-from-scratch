import torch
from typing import Tuple

from src.tokenizer import CharTokenizer


def load_text(path: str) -> str:
    """Load raw text data from a txt file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
def build_dataset(
        text: str,
        tokenizer: CharTokenizer,
        train_ratio: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert raw text into token ids and split it into train / validation sets.
    
    Returns:
        train_data: shape [num_train_tokens]
        val_data: shape [num_val_tokens]
    """
    ids = tokenizer.encode(text)
    data = torch.tensor(ids, dtype=torch.long)  # [N]

    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]    # [N_train], N_train ~= train_ratio * N
    val_data = data[split_idx:]  # [N_val], N_val ~= (1 - train_ratio) * N

    return train_data, val_data

def get_batch(
        data: torch.Tensor,
        batch_size: int,
        block_size: int,
        device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly sample a batch of input-target sequences.
    
    Example:
        If the original token sequence is:
        [0, 1, 2, 3, 4, 5]
        
        and block_size = 4,
        
        x = [0, 1, 2, 3]
        y = [1, 2, 3, 4]
        
    Shapes:
        x: [batch_size, block_size]
        y: [batch_size, block_size]
        """
    if len(data) <= block_size:
        raise ValueError(
            f"Data length must be larger than block_size."
            f"Got len(data)={len(data)}, block_size={block_size}"
        )
    
    ix = torch.randint(0, len(data)-block_size, (batch_size,))  # [B]

    x = torch.stack([data[i: i + block_size] for i in ix])  # [B, T]
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])  # [B, T]

    return x.to(device), y.to(device)