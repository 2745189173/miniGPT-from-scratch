import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.bpe_tokenizer import BPETokenizer


def main():
    training_text = (
        "hello hello world! "
        "hello hello transformer! "
    ) * 20

    test_text = "hello transformer! 你好"

    tokenizer = BPETokenizer()
    tokenizer.train(
        text=training_text,
        target_vocab_size=280,
    )

    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print("vocab size:", tokenizer.vocab_size)
    print("UTF-8 byte count:", len(test_text.encode("utf-8")))
    print("BPE token count:", len(encoded))
    print("encoded:", encoded)
    print("decoded:", decoded)

    assert decoded == test_text, (
        "BPE encode/decode should preserve the original text."
    )

    assert all(
        0 <= token_id < tokenizer.vocab_size
        for token_id in encoded
    ), "All token ids should be inside the vocabulary."

    assert len(encoded) < len(test_text.encode("utf-8")), (
        "BPE should compress repeated training patterns."
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        tokenizer_path = (
            Path(temp_dir) / "tokenizer.json"
        )

        tokenizer.save(str(tokenizer_path))
        loaded_tokenizer = BPETokenizer.load(
            str(tokenizer_path)
        )

        loaded_ids = loaded_tokenizer.encode(test_text)
        loaded_text = loaded_tokenizer.decode(
            loaded_ids
        )

        assert loaded_ids == encoded, (
            "Loaded tokenizer should reproduce token ids."
        )
        assert loaded_text == test_text, (
            "Loaded tokenizer should preserve round trips."
        )

    print(
        "\ncheck passed: BPE training, compression, "
        "round trip, save, and load are correct."
    )


if __name__ == "__main__":
    main()