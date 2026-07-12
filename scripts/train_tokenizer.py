import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.bpe_tokenizer import BPETokenizer
from src.dataset import load_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/tiny.yaml",
        help="Config path relative to the project root.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with open(config_path, encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    tokenizer_config = config_data["tokenizer"]
    if tokenizer_config["type"] != "byte_bpe":
        raise ValueError("Configured tokenizer is not byte_bpe.")

    text = load_text(config_data["data"]["input_path"])
    target_vocab_size = tokenizer_config["vocab_size"]
    output_path = PROJECT_ROOT / tokenizer_config["path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = BPETokenizer()
    tokenizer.train(
        text=text,
        target_vocab_size=target_vocab_size,
    )
    tokenizer.save(str(output_path))

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    if decoded != text:
        raise RuntimeError("Trained BPE tokenizer failed round trip.")

    byte_count = len(text.encode("utf-8"))
    print("characters:", len(text))
    print("UTF-8 bytes:", byte_count)
    print("BPE tokens:", len(encoded))
    print("vocab size:", tokenizer.vocab_size)
    print("bytes per token:", f"{byte_count / len(encoded):.4f}")
    print("saved tokenizer:", output_path)


if __name__ == "__main__":
    main()
