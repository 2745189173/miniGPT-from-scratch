import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset import load_text
from src.tokenizer import CharTokenizer
from src.tokenizer_factory import build_tokenizer


def main():
    with open(
        PROJECT_ROOT / "configs" / "tiny.yaml",
        encoding="utf-8",
    ) as file:
        config_data = yaml.safe_load(file)

    text = load_text(config_data["data"]["input_path"])
    char_tokenizer = CharTokenizer(text)
    bpe_tokenizer = build_tokenizer(config_data, text, PROJECT_ROOT)

    char_count = len(text)
    byte_count = len(text.encode("utf-8"))
    char_tokens = len(char_tokenizer.encode(text))
    bpe_tokens = len(bpe_tokenizer.encode(text))
    block_size = config_data["model"]["block_size"]

    result = {
        "text_characters": char_count,
        "text_utf8_bytes": byte_count,
        "char_vocab_size": char_tokenizer.vocab_size,
        "bpe_vocab_size": bpe_tokenizer.vocab_size,
        "char_token_count": char_tokens,
        "bpe_token_count": bpe_tokens,
        "token_count_reduction": 1 - bpe_tokens / char_tokens,
        "bpe_bytes_per_token": byte_count / bpe_tokens,
        "char_context_characters": block_size,
        "estimated_bpe_context_characters": (block_size * char_count / bpe_tokens),
    }

    output_path = PROJECT_ROOT / "experiments" / "tokenizer_comparison.json"
    with open(output_path, "w", encoding="utf-8", newline="\n") as file:
        json.dump(result, file, indent=2)

    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("saved comparison:", output_path)


if __name__ == "__main__":
    main()
