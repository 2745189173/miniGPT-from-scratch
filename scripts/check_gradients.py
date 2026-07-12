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

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("device:", device)
    print("total parameters:", total_params)
    print("trainable parameters:", trainable_params)

    assert total_params > 0, "Model should contain parameters."
    assert trainable_params == total_params, (
        "All model parameters should currently be trainable."
    )

    x, y = get_batch(
        data=train_data,
        batch_size=4,
        block_size=config.block_size,
        device=device,
    )

    model.train()
    model.zero_grad(set_to_none=True)

    _, loss = model(x, y)

    assert loss is not None, (
        "Loss is required before backward propagation."
    )
    assert torch.isfinite(loss), (
        "Loss should be finite before backward propagation."
    )

    loss.backward()

    missing_gradients = []
    non_finite_gradients = []
    gradients = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if parameter.grad is None:
            missing_gradients.append(name)
            continue

        if not torch.isfinite(parameter.grad).all():
            non_finite_gradients.append(name)

        gradients.append(parameter.grad)

    assert not missing_gradients, (
        f"Parameters without gradients: {missing_gradients}"
    )
    assert not non_finite_gradients, (
        f"Parameters with non-finite gradients: {non_finite_gradients}"
    )

    global_grad_norm = torch.sqrt(
        sum(gradient.pow(2).sum() for gradient in gradients)
    )

    print("loss:", loss.item())
    print("global gradient norm:", global_grad_norm.item())

    assert torch.isfinite(global_grad_norm), (
        "Global gradient norm should be finite."
    )
    assert global_grad_norm.item() > 0, (
        "Global gradient norm should be greater than zero."
    )

    print(
        "\ncheck passed: parameter count and gradient flow are correct."
    )


if __name__ == "__main__":
    main()
