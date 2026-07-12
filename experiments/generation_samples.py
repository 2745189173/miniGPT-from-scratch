import json
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

    with open(config_path, encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    run_name = config_data["experiment"]["run_name"]
    train_config = config_data["train"]
    generate_config = config_data["generate"]

    requested_device = train_config["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    else:
        device = requested_device

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
        character for character in prompt if character not in tokenizer.stoi
    ]

    if unknown_characters:
        raise ValueError(f"Prompt contains unknown characters: {unknown_characters}")

    prompt_ids = tokenizer.encode(prompt)
    temperatures = [0.5, 0.8, 1.1, 1.4]
    fixed_top_k = generate_config["top_k"]
    seed = config_data["seed"]

    results = []

    for temperature in temperatures:
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        idx = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=device,
        )

        generated_ids = generate(
            model=model,
            idx=idx,
            max_new_tokens=generate_config["max_new_tokens"],
            temperature=temperature,
            top_k=fixed_top_k,
        )

        generated_text = tokenizer.decode(generated_ids[0].tolist())

        result = {
            "temperature": temperature,
            "top_k": fixed_top_k,
            "text": generated_text,
        }
        results.append(result)

        print("\n" + "=" * 70)
        print(f"temperature={temperature}, top_k={fixed_top_k}")
        print("=" * 70)
        print(generated_text)

    output_dir = PROJECT_ROOT / "experiments" / "generation_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "e004_temperature_comparison.json"

    experiment_data = {
        "experiment": "E004",
        "checkpoint_run": run_name,
        "checkpoint_step": checkpoint["step"],
        "prompt": prompt,
        "seed": seed,
        "max_new_tokens": generate_config["max_new_tokens"],
        "fixed_top_k": fixed_top_k,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            experiment_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nresults saved to:", output_path)


if __name__ == "__main__":
    main()
