import json
from typing import List


class CharTokenizer:
    """
    A simple character-level tokenizer.

    It maps each unique character in the training text to an interger token id.
    This is not how production LLM tokenizers work, but it is ideal for learning
    the core language modeling pipeline.
    """

    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> List[int]:
        """Convert a string into a list of token ids."""
        return [self.stoi[ch] for ch in text]
    
    def decode(self, ids: List[int]) -> str:
        """Convert a list of token ids into a string"""
        return "".join([self.itos[i] for i in ids])
    
    def save(self, path: str) -> None:
        """Save tokenizer vocabulary to a JSON file"""
        data = {
            "stoi": self.stoi,
            "itos": {str(k):v for k, v in self.itos.items()},
            "vocab_size": self.vocab_size,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load tokenizer vocabulary from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls.__new__(cls)
        tokenizer.stoi = data["stoi"]
        tokenizer.itos = {int(k): v for k, v in data["itos"].items()}
        tokenizer.vocab_size = data["vocab_size"]

        return tokenizer