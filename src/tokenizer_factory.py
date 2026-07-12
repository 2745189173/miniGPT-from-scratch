from pathlib import Path
from typing import Any

from src.bpe_tokenizer import BPETokenizer
from src.tokenizer import CharTokenizer, Tokenizer


def build_tokenizer(
    config_data: dict[str, Any],
    text: str,
    project_root: Path,
) -> Tokenizer:
    tokenizer_config = config_data.get(
        "tokenizer",
        {"type": "char"},
    )
    tokenizer_type = tokenizer_config.get("type", "char")

    if tokenizer_type == "char":
        return CharTokenizer(text)

    if tokenizer_type == "byte_bpe":
        tokenizer_path = project_root / tokenizer_config["path"]
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"BPE tokenizer not found: {tokenizer_path}. "
                "Run python scripts/train_tokenizer.py first."
            )
        tokenizer = BPETokenizer.load(str(tokenizer_path))
        expected_size = tokenizer_config.get("vocab_size")
        if expected_size is not None and tokenizer.vocab_size != expected_size:
            raise ValueError(
                "Configured BPE vocab size does not match "
                f"the saved tokenizer: {expected_size} != "
                f"{tokenizer.vocab_size}."
            )
        return tokenizer

    raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")


def tokenizer_from_state(state: dict[str, Any]) -> Tokenizer:
    tokenizer_type = state.get("tokenizer_type")

    if tokenizer_type == "byte_bpe":
        return BPETokenizer.from_state(state)

    if tokenizer_type == "char" or "stoi" in state:
        return CharTokenizer.from_state(state)

    raise ValueError("Unsupported tokenizer checkpoint state.")
