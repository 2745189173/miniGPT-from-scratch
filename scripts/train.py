import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset import build_dataset, get_batch, load_text
from src.model import GPTConfig, GPTLanguageModel
from src.tokenizer import CharTokenizer


@torch.no_grad()
def estimate_loss(
    model,
    train_data,
    val_data,
    batch_size,
    block_size,
    eval_iters,
    device,
):
    model.eval()
    losses = {}

    for split_name, data in {
        "train": train_data,
        "val": val_data,
    }.items():
        split_losses = torch.zeros(eval_iters)

        for index in range(eval_iters):
            x, y = get_batch(
                data=data,
                batch_size=batch_size,
                block_size=block_size,
                device=device,
            )

            _, loss = model(x, y)

            if loss is None:
                raise RuntimeError(
                    "Model did not return loss during evaluation."
                )

            split_losses[index] = loss.item()

        losses[split_name] = split_losses.mean().item()

    model.train()
    return losses

def load_resume_state(
    model,
    optimizer,
    model_config,
    resume_run,
    device,
):
    checkpoint_path = (
        PROJECT_ROOT
        / "experiments"
        / "checkpoints"
        / f"{resume_run}.pt"
    )

    history_path = (
        PROJECT_ROOT
        / "experiments"
        / "loss_curves"
        / f"{resume_run}.json"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {checkpoint_path}"
        )

    if not history_path.exists():
        raise FileNotFoundError(
            f"Resume loss history not found: {history_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    expected_model_config = asdict(model_config)

    if checkpoint["model_config"] != expected_model_config:
        raise ValueError(
            "Resume checkpoint model configuration does not "
            "match the current model configuration."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    with open(history_path, "r", encoding="utf-8") as file:
        loss_history = json.load(file)

    if "cpu_rng_state" in checkpoint:
        torch.set_rng_state(
            checkpoint["cpu_rng_state"].cpu()
        )
    else:
        print(
            "warning: checkpoint has no CPU RNG state; "
            "continuation will not be bitwise reproducible."
        )

    if (
        device == "cuda"
        and checkpoint.get("cuda_rng_state_all") is not None
    ):
        cuda_rng_states = [
            state.cpu()
            for state in checkpoint["cuda_rng_state_all"]
        ]
        torch.cuda.set_rng_state_all(cuda_rng_states)

    print("resumed from:", checkpoint_path)
    print("resume step:", checkpoint["step"])
    print("previous best val loss:", checkpoint["val_loss"])

    return checkpoint, loss_history

def main():
    config_path = PROJECT_ROOT / "configs" / "tiny.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    seed = config_data["seed"]
    run_name = config_data["experiment"]["run_name"]
    data_config = config_data["data"]
    model_config = config_data["model"]
    train_config = config_data["train"]
    resume_from = train_config.get("resume_from")

    if not run_name or not all(
        character.isalnum() or character in {"-", "_"}
        for character in run_name
    ):
        raise ValueError(
            "experiment.run_name may contain only letters, numbers, "
            "hyphens, and underscores."
        )

    torch.manual_seed(seed)

    requested_device = train_config["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    else:
        device = requested_device

    text = load_text(data_config["input_path"])
    tokenizer = CharTokenizer(text)
    train_data, val_data = build_dataset(text, tokenizer)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=model_config["block_size"],
        n_embd=model_config["n_embd"],
        num_heads=model_config["n_head"],
        num_layers=model_config["n_layer"],
        dropout=model_config["dropout"],
    )

    model = GPTLanguageModel(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    checkpoint_dir = PROJECT_ROOT / "experiments" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = checkpoint_dir / f"{run_name}.pt"
    start_step = 0
    best_val_loss = float("inf")
    best_step = -1
    loss_history = []

    if resume_from is not None:
        resume_checkpoint, loss_history = load_resume_state(
            model=model,
            optimizer=optimizer,
            model_config=config,
            resume_run=resume_from,
            device=device,
        )

        start_step = resume_checkpoint["step"]
        best_step = resume_checkpoint["step"]
        best_val_loss = resume_checkpoint["val_loss"]

        resume_checkpoint["run_name"] = run_name
        resume_checkpoint["run_config"] = config_data
        resume_checkpoint["cpu_rng_state"] = (
            torch.get_rng_state()
        )
        resume_checkpoint["cuda_rng_state_all"] = (
            torch.cuda.get_rng_state_all()
            if device == "cuda"
            else None
        )

        torch.save(
            resume_checkpoint,
            best_checkpoint_path,
        )

    print("run name:", run_name)
    print("start step:", start_step)
    print("target step:", train_config["max_iters"])
    print("device:", device)
    print("vocab size:", tokenizer.vocab_size)
    print("train tokens:", len(train_data))
    print("validation tokens:", len(val_data))
    print("parameters:", total_params)

    max_iters = train_config["max_iters"]
    eval_interval = train_config["eval_interval"]

    for step in range(start_step, max_iters + 1):
        is_resume_boundary = (
            resume_from is not None
            and step == start_step
        )

        should_evaluate = (
            step % eval_interval == 0
            or step == max_iters
        )

        if should_evaluate and not is_resume_boundary:
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                batch_size=train_config["batch_size"],
                block_size=config.block_size,
                eval_iters=train_config["eval_iters"],
                device=device,
            )

            print(
                f"step {step:4d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )
            loss_history.append(
                {
                    "step": step,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                }
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                best_step = step

                checkpoint = {
                    "run_name": run_name,
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": asdict(config),
                    "tokenizer": {
                        "stoi": tokenizer.stoi,
                        "itos": tokenizer.itos,
                        "vocab_size": tokenizer.vocab_size,
                    },
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "run_config": config_data,
                    "cpu_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": (
                        torch.cuda.get_rng_state_all()
                        if device == "cuda"
                        else None
                    ),
                }

                torch.save(checkpoint, best_checkpoint_path)

                print(
                    f"saved new best checkpoint: "
                    f"step {step}, val loss {best_val_loss:.4f}"
                )

        if step == max_iters:
            break

        x, y = get_batch(
            data=train_data,
            batch_size=train_config["batch_size"],
            block_size=config.block_size,
            device=device,
        )

        _, loss = model(x, y)

        if loss is None:
            raise RuntimeError(
                "Model did not return loss during training."
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    loss_curve_dir = PROJECT_ROOT / "experiments" / "loss_curves"
    loss_curve_dir.mkdir(parents=True, exist_ok=True)

    loss_history_path = loss_curve_dir / f"{run_name}.json"

    with open(loss_history_path, "w", encoding="utf-8") as file:
        json.dump(loss_history, file, indent=2)

    print("loss history:", loss_history_path)

    print("\ntraining completed.")
    print("best step:", best_step)
    print("best validation loss:", f"{best_val_loss:.4f}")
    print("checkpoint:", best_checkpoint_path)


if __name__ == "__main__":
    main()
