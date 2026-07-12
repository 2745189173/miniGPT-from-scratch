import argparse
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.generation import generate
from src.model import GPTConfig, GPTLanguageModel
from src.tokenizer_factory import tokenizer_from_state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/tiny.yaml",
        help="Config path relative to the project root.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with open(config_path, encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    train_config = config_data["train"]
    generate_config = config_data["generate"]
    run_name = config_data["experiment"]["run_name"]

    requested_device = train_config["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    else:
        device = requested_device

    torch.manual_seed(config_data["seed"])

    checkpoint_path = PROJECT_ROOT / "experiments" / "checkpoints" / f"{run_name}.pt"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model_config = GPTConfig(**checkpoint["model_config"])
    model = GPTLanguageModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = tokenizer_from_state(checkpoint["tokenizer"])

    prompt = generate_config["prompt"]

    try:
        prompt_ids = tokenizer.encode(prompt)
    except (KeyError, UnicodeEncodeError) as error:
        raise ValueError(
            f"Prompt cannot be encoded by this tokenizer: {prompt!r}"
        ) from error
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

    generated_text = tokenizer.decode(generated_ids[0].tolist())

    print("run name:", run_name)
    print("checkpoint step:", checkpoint["step"])
    print("checkpoint val loss:", checkpoint["val_loss"])
    print("device:", device)
    print("\ngenerated text:\n")
    print(generated_text)


if __name__ == "__main__":
    main()
