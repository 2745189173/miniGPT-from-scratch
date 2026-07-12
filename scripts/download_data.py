import argparse
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "input.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if OUTPUT_PATH.exists() and not args.force:
        print("dataset already exists:", OUTPUT_PATH)
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, OUTPUT_PATH)
    print("downloaded dataset:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
