import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset import build_dataset, get_batch, load_text
from src.model import GPTConfig, GPTLanguageModel
from src.tokenizer import CharTokenizer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = load_text("data/raw/input.txt")
    tokenizer = CharTokenizer(text)
    train_data, _ = build_dataset(text, tokenizer)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=8,
        n_embd=32,
        num_heads=4,
        num_layers=2,
        activation="gelu",
        tie_weights=True,
        dropout=0.1,
    )
    model = GPTLanguageModel(config).to(device)
    print("num Transformer blocks:", len(model.blocks))

    assert model.lm_head.weight is model.token_embedding.weight, (
        "LM head and token embedding should share one Parameter."
    )
    assert (
        model.lm_head.weight.data_ptr()
        == model.token_embedding.weight.data_ptr()
    ), "Tied weights should share the same tensor storage."

    assert len(model.blocks) == config.num_layers, (
        "Model should contain config.num_layers Transformer blocks."
    )

    x, y = get_batch(
        data=train_data,
        batch_size=4,
        block_size=config.block_size,
        device=device,
    )

    logits, loss = model(x, y)

    print("device:", device)
    print("vocab_size:", tokenizer.vocab_size)
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("logits shape:", logits.shape)
    print("loss:", loss.item())

    assert logits.shape == (
        x.shape[0],
        x.shape[1],
        tokenizer.vocab_size,
    ), "Logits should have shape [batch_size, block_size, vocab_size]."
    assert loss is not None, "Loss should be returned when targets are provided."
    assert torch.isfinite(loss), "Loss should be a finite scalar."

    logits_without_targets, loss_without_targets = model(x)
    assert logits_without_targets.shape == logits.shape, (
        "Logits shape should be the same whether targets are provided or not."
    )
    assert loss_without_targets is None, (
        "Loss should be None when targets are not provided."
    )

    print(
        "\ncheck passed: Transformer model forward shape and loss are correct."
    )


if __name__ == "__main__":
    main()
