import json
from typing import Any, Protocol


class Tokenizer(Protocol):
    vocab_size: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    def save(self, path: str) -> None: ...

    def to_state(self) -> dict[str, Any]: ...


class CharTokenizer:
    """
    A simple character-level tokenizer.

    It maps each unique character in the training text to an integer token id.
    This is not how production LLM tokenizers work, but it is ideal for learning
    the core language modeling pipeline.
    """

    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        """Convert a string into a list of token ids."""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Convert a list of token ids into a string"""
        return "".join([self.itos[i] for i in ids])

    def save(self, path: str) -> None:
        """Save tokenizer vocabulary to a JSON file"""
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(
                self.to_state(),
                f,
                ensure_ascii=False,
                indent=2,
            )

    def to_state(self) -> dict[str, Any]:
        return {
            "tokenizer_type": "char",
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_state(cls, data: dict[str, Any]):
        tokenizer = cls.__new__(cls)
        tokenizer.stoi = data["stoi"]
        tokenizer.itos = {int(k): v for k, v in data["itos"].items()}
        tokenizer.vocab_size = data["vocab_size"]
        return tokenizer

    @classmethod
    def load(cls, path: str):
        """Load tokenizer vocabulary from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_state(data)
