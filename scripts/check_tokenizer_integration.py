import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset import build_dataset, load_text
from src.tokenizer_factory import build_tokenizer, tokenizer_from_state


def main():
    with open(
        PROJECT_ROOT / "configs" / "tiny.yaml",
        "r",
        encoding="utf-8",
    ) as file:
        config_data = yaml.safe_load(file)

    text = load_text(config_data["data"]["input_path"])
    tokenizer = build_tokenizer(config_data, text, PROJECT_ROOT)
    restored = tokenizer_from_state(tokenizer.to_state())

    sample = "First Citizen:\nSpeak, speak."
    token_ids = tokenizer.encode(sample)

    assert restored.encode(sample) == token_ids
    assert restored.decode(token_ids) == sample
    assert restored.vocab_size == tokenizer.vocab_size

    train_data, val_data = build_dataset(text, tokenizer)
    assert len(train_data) > 0 and len(val_data) > 0
    assert int(train_data.max()) < tokenizer.vocab_size
    assert int(val_data.max()) < tokenizer.vocab_size

    print("tokenizer type:", type(tokenizer).__name__)
    print("vocab size:", tokenizer.vocab_size)
    print("sample token count:", len(token_ids))
    print("train tokens:", len(train_data))
    print("validation tokens:", len(val_data))
    print("\ncheck passed: tokenizer state and dataset integration work.")


if __name__ == "__main__":
    main()
