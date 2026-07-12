import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOSS_CURVE_DIR = PROJECT_ROOT / "experiments" / "loss_curves"
CONFIG_PATH = PROJECT_ROOT / "configs" / "tiny.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot train and validation loss for one experiment."
    )
    parser.add_argument(
        "--run-name",
        help="Experiment run name. Defaults to configs/tiny.yaml.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.run_name is None:
        with open(CONFIG_PATH, encoding="utf-8") as file:
            config_data = yaml.safe_load(file)
        run_name = config_data["experiment"]["run_name"]
    else:
        run_name = args.run_name

    history_path = LOSS_CURVE_DIR / f"{run_name}.json"
    output_path = LOSS_CURVE_DIR / f"{run_name}.png"

    with open(history_path, encoding="utf-8") as file:
        history = json.load(file)

    if not history:
        raise ValueError("Loss history is empty.")

    steps = [record["step"] for record in history]
    train_losses = [record["train_loss"] for record in history]
    val_losses = [record["val_loss"] for record in history]

    best_index = min(
        range(len(val_losses)),
        key=lambda index: val_losses[index],
    )

    best_step = steps[best_index]
    best_val_loss = val_losses[best_index]

    figure, axis = plt.subplots(figsize=(9, 5))

    axis.plot(
        steps,
        train_losses,
        marker="o",
        label="Train loss",
    )
    axis.plot(
        steps,
        val_losses,
        marker="o",
        label="Validation loss",
    )

    axis.scatter(
        best_step,
        best_val_loss,
        color="red",
        zorder=3,
        label=f"Best validation: step {best_step}",
    )

    axis.axvline(
        best_step,
        color="red",
        linestyle="--",
        alpha=0.4,
    )

    axis.set_title(f"miniGPT Loss - {run_name}")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Cross-entropy loss")
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    print("best step:", best_step)
    print("best validation loss:", f"{best_val_loss:.4f}")
    print("saved loss curve:", output_path)


if __name__ == "__main__":
    main()
