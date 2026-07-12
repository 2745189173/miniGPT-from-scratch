import json
from collections import Counter
from typing import Dict, List, Tuple


Pair = Tuple[int, int]


def merge_pair(
    token_ids: List[int],
    pair: Pair,
    new_token_id: int,
) -> List[int]:
    merged_ids = []
    index = 0

    while index < len(token_ids):
        can_merge = (
            index < len(token_ids) - 1
            and token_ids[index] == pair[0]
            and token_ids[index + 1] == pair[1]
        )

        if can_merge:
            merged_ids.append(new_token_id)
            index += 2
        else:
            merged_ids.append(token_ids[index])
            index += 1

    return merged_ids


class BPETokenizer:
    """
    Educational byte-level BPE tokenizer.

    Token ids 0-255 represent raw bytes.
    New merged tokens start at id 256.
    """

    def __init__(self):
        self.merges: List[Pair] = []
        self.vocab: Dict[int, bytes] = {
            token_id: bytes([token_id])
            for token_id in range(256)
        }
        self.vocab_size = 256

    def train(
        self,
        text: str,
        target_vocab_size: int,
    ) -> None:
        if target_vocab_size < 256:
            raise ValueError(
                "Byte-level BPE vocab size must be at least 256."
            )

        token_ids = list(text.encode("utf-8"))
        num_merges = target_vocab_size - 256

        self.merges = []
        self.vocab = {
            token_id: bytes([token_id])
            for token_id in range(256)
        }

        for merge_index in range(num_merges):
            if len(token_ids) < 2:
                break

            pair_counts = Counter(
                zip(token_ids, token_ids[1:])
            )

            if not pair_counts:
                break

            best_pair = max(
                pair_counts,
                key=lambda pair: (
                    pair_counts[pair],
                    -pair[0],
                    -pair[1],
                ),
            )

            new_token_id = 256 + merge_index

            token_ids = merge_pair(
                token_ids=token_ids,
                pair=best_pair,
                new_token_id=new_token_id,
            )

            self.merges.append(best_pair)
            self.vocab[new_token_id] = (
                self.vocab[best_pair[0]]
                + self.vocab[best_pair[1]]
            )

        self.vocab_size = 256 + len(self.merges)

    def encode(self, text: str) -> List[int]:
        token_ids = list(text.encode("utf-8"))

        for merge_index, pair in enumerate(self.merges):
            new_token_id = 256 + merge_index
            token_ids = merge_pair(
                token_ids=token_ids,
                pair=pair,
                new_token_id=new_token_id,
            )

        return token_ids

    def decode(self, ids: List[int]) -> str:
        unknown_ids = [
            token_id
            for token_id in ids
            if token_id not in self.vocab
        ]

        if unknown_ids:
            raise ValueError(
                f"Unknown BPE token ids: {unknown_ids}"
            )

        byte_sequence = b"".join(
            self.vocab[token_id]
            for token_id in ids
        )

        return byte_sequence.decode(
            "utf-8",
            errors="replace",
        )

    def save(self, path: str) -> None:
        data = {
            "tokenizer_type": "byte_bpe",
            "vocab_size": self.vocab_size,
            "merges": [
                [left_id, right_id]
                for left_id, right_id in self.merges
            ],
        }

        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if data.get("tokenizer_type") != "byte_bpe":
            raise ValueError(
                "File does not contain a byte-level BPE tokenizer."
            )

        tokenizer = cls()
        tokenizer.merges = [
            (left_id, right_id)
            for left_id, right_id in data["merges"]
        ]

        for merge_index, pair in enumerate(
            tokenizer.merges
        ):
            new_token_id = 256 + merge_index
            tokenizer.vocab[new_token_id] = (
                tokenizer.vocab[pair[0]]
                + tokenizer.vocab[pair[1]]
            )

        tokenizer.vocab_size = len(tokenizer.vocab)

        return tokenizer