import torch

from src.model import GPTLanguageModel


@torch.no_grad()
def generate(
    model: GPTLanguageModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")

    model.eval()

    for _ in range(max_new_tokens):
        idx_context = idx[:, -model.config.block_size :]

        logits, _ = model(idx_context)
        logits = logits[:, -1, :]
        logits = logits / temperature

        if top_k is not None:
            effective_top_k = min(top_k, logits.shape[-1])
            top_values, _ = torch.topk(logits, effective_top_k)
            threshold = top_values[:, [-1]]

            logits = logits.masked_fill(
                logits < threshold,
                float("-inf"),
            )

        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        idx = torch.cat((idx, next_token), dim=1)

    return idx
