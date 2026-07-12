import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset import build_dataset, get_batch, load_text
from src.tokenizer import CharTokenizer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = load_text("data/raw/input.txt")
    tokenizer = CharTokenizer(text)

    train_data, val_data = build_dataset(text, tokenizer)

    x, y = get_batch(
        data=train_data,
        batch_size=4,
        block_size=8,
        device=device,
    )

    print("device:", device)
    print("vocab_size:", tokenizer.vocab_size)
    print("train_data shape:", train_data.shape)
    print("val_data shape:", val_data.shape)
    print("x shape:", x.shape)
    print("y shape:", y.shape)

    print("\nx[0]:", x[0])
    print("y[0]:", y[0])

    print("\ndecode x[0]:", repr(tokenizer.decode(x[0].tolist())))
    print("decode y[0]:", repr(tokenizer.decode(y[0].tolist())))

    assert torch.equal(x[:, 1:], y[:, :-1]), (
        "Batch construction error: y is not x shifted by one position."
    )

    print("\ncheck passed: y is x shifted by one position.")


if __name__ == "__main__":
    main()
