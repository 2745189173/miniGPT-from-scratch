import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.generation import generate
from src.model import GPTConfig, GPTLanguageModel
from src.tokenizer import CharTokenizer


def main():
    config_path = PROJECT_ROOT / "configs" / "tiny.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    train_config = config_data["train"]
    generate_config = config_data["generate"]

    requested_device = train_config["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    else:
        device = requested_device

    torch.manual_seed(config_data["seed"])

    checkpoint_path = (
        PROJECT_ROOT / generate_config["checkpoint_path"]
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model_config = GPTConfig(**checkpoint["model_config"])
    model = GPTLanguageModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer_data = checkpoint["tokenizer"]

    tokenizer = CharTokenizer.__new__(CharTokenizer)
    tokenizer.stoi = tokenizer_data["stoi"]
    tokenizer.itos = {
        int(token_id): character
        for token_id, character in tokenizer_data["itos"].items()
    }
    tokenizer.vocab_size = tokenizer_data["vocab_size"]

    prompt = generate_config["prompt"]

    unknown_characters = [
        character
        for character in prompt
        if character not in tokenizer.stoi
    ]

    if unknown_characters:
        raise ValueError(
            f"Prompt contains unknown characters: "
            f"{unknown_characters}"
        )

    prompt_ids = tokenizer.encode(prompt)
    idx = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    generated_ids = generate(
        model=model,
        idx=idx,
        max_new_tokens=generate_config["max_new_tokens"],
        temperature=generate_config["temperature"],
        top_k=generate_config["top_k"],
    )

    generated_text = tokenizer.decode(
        generated_ids[0].tolist()
    )

    print("checkpoint step:", checkpoint["step"])
    print("checkpoint val loss:", checkpoint["val_loss"])
    print("device:", device)
    print("\ngenerated text:\n")
    print(generated_text)


if __name__ == "__main__":
    main()