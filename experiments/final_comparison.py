import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.model import GPTConfig, GPTLanguageModel


def load_checkpoint(run_name: str):
    return torch.load(
        PROJECT_ROOT / "experiments" / "checkpoints" / f"{run_name}.pt",
        map_location="cpu",
        weights_only=False,
    )


def parameter_count(checkpoint) -> int:
    config_data = checkpoint["model_config"].copy()
    config_data.setdefault("activation", "relu")
    config_data.setdefault("tie_weights", False)
    model = GPTLanguageModel(GPTConfig(**config_data))
    model.load_state_dict(checkpoint["model_state_dict"])
    return sum(parameter.numel() for parameter in model.parameters())


def main():
    with open(
        PROJECT_ROOT / "experiments" / "tokenizer_comparison.json",
        encoding="utf-8",
    ) as file:
        tokenizer_metrics = json.load(file)

    char_checkpoint = load_checkpoint("e010_shakespeare_gelu")
    bpe_checkpoint = load_checkpoint("e012_shakespeare_bpe512")

    characters_per_bpe_token = (
        tokenizer_metrics["text_characters"] / tokenizer_metrics["bpe_token_count"]
    )
    char_bpc = char_checkpoint["val_loss"] / math.log(2)
    bpe_bpc = bpe_checkpoint["val_loss"] / math.log(2) / characters_per_bpe_token

    result = {
        "character_baseline": {
            "run": "e010_shakespeare_gelu",
            "vocab_size": 65,
            "parameters": parameter_count(char_checkpoint),
            "best_step": char_checkpoint["step"],
            "validation_loss_per_token": char_checkpoint["val_loss"],
            "estimated_bits_per_character": char_bpc,
        },
        "bpe_model": {
            "run": "e012_shakespeare_bpe512",
            "vocab_size": 512,
            "parameters": parameter_count(bpe_checkpoint),
            "best_step": bpe_checkpoint["step"],
            "validation_loss_per_token": bpe_checkpoint["val_loss"],
            "estimated_bits_per_character": bpe_bpc,
        },
        "token_count_reduction": tokenizer_metrics["token_count_reduction"],
        "context_coverage_multiplier": (
            tokenizer_metrics["estimated_bpe_context_characters"]
            / tokenizer_metrics["char_context_characters"]
        ),
        "estimated_bpc_improvement": 1 - bpe_bpc / char_bpc,
    }

    output_path = PROJECT_ROOT / "experiments" / "final_comparison.json"
    with open(output_path, "w", encoding="utf-8", newline="\n") as file:
        json.dump(result, file, indent=2)

    print(json.dumps(result, indent=2))
    print("saved final comparison:", output_path)


if __name__ == "__main__":
    main()
