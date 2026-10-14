"""Training script for Track A CoreModel on preprocessed stroke datasets."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data.loader import StrokeDataset, collate_stroke_batch
from .dataset import LineStore
from .models.core import CoreConfig, CoreModel, build, count_params
from .text_codec import VOCAB_SIZE
from .tokenizer import StrokeTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train VibeFont Track A CoreModel")
    parser.add_argument("--data", default="data/processed/synthetic", help="Path to processed dataset")
    parser.add_argument("--out-dir", default="checkpoints/core_xs", help="Directory to save model checkpoints")
    parser.add_argument("--preset", default="xs", choices=["xs", "s", "m"], help="Model size preset")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=4e-4, help="Peak learning rate")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Linear warmup steps")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--val-split", type=float, default=0.05, help="Validation set split fraction")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def get_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float = 1e-5) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    decay_ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print(f"Loading dataset from {args.data}...")
    store = LineStore.load(args.data)
    tokenizer = StrokeTokenizer()
    dataset = StrokeDataset(store, tokenizer)

    # Save tokenizer for inference
    tokenizer.save(out_dir / "tokenizer.json")

    # Train / Val split
    n_total = len(dataset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    indices = np.random.permutation(n_total)
    train_ds = Subset(dataset, indices[:n_train])
    val_ds = Subset(dataset, indices[n_train:])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_stroke_batch,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_stroke_batch,
        num_workers=0,
    )
    print(f"Dataset: {n_train} train lines, {n_val} val lines, {len(dataset.writer_map)} writers")

    # 2. Build model
    n_writers = max(len(dataset.writer_map), 1)
    model = build(
        args.preset,
        vocab_size=VOCAB_SIZE,
        n_dx=tokenizer.sizes["dx"],
        n_dy=tokenizer.sizes["dy"],
        n_dt=tokenizer.sizes["dt"],
        n_writers=n_writers,
    ).to(device)

    print(f"Model ({args.preset}): {count_params(model):,} parameters")

    # Save config
    (out_dir / "config.json").write_text(json.dumps(model.c.to_dict(), indent=2), encoding="utf-8")

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    global_step = 0
    best_val_loss = float("inf")

    # 4. Training loop
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        train_losses = []

        for step, batch in enumerate(train_loader):
            lr = get_lr(global_step, total_steps, args.warmup_steps, args.lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Move batch to device
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            optimizer.zero_grad()
            losses = model(batch_dev)
            loss = losses["loss"]
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            global_step += 1

            if (step + 1) % 25 == 0 or (step + 1) == len(train_loader):
                pen_l = losses["pen"].item()
                dx_l = losses["dx"].item()
                dy_l = losses["dy"].item()
                dt_l = losses["dt"].item()
                print(
                    f"Epoch {epoch:02d}/{args.epochs:02d} [{step+1:03d}/{len(train_loader):03d}] "
                    f"Loss: {loss.item():.4f} (pen:{pen_l:.3f} dx:{dx_l:.3f} dy:{dy_l:.3f} dt:{dt_l:.3f}) "
                    f"LR: {lr:.2e}",
                    flush=True,
                )

        train_loss_avg = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                losses = model(batch_dev)
                val_losses.append(losses["loss"].item())

        val_loss_avg = np.mean(val_losses)
        dt_epoch = time.perf_counter() - t0
        print(
            f"=== Epoch {epoch:02d} done in {dt_epoch:.1f}s | "
            f"Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} ===",
            flush=True,
        )

        # Save checkpoint
        is_best = val_loss_avg < best_val_loss
        if is_best:
            best_val_loss = val_loss_avg
            ckpt_path = out_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": val_loss_avg,
                    "config": model.c.to_dict(),
                    "model_state": model.state_dict(),
                },
                ckpt_path,
            )
            # Also save as default model.pt
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": val_loss_avg,
                    "config": model.c.to_dict(),
                    "model_state": model.state_dict(),
                },
                out_dir / "model.pt",
            )
            print(f"--> Saved new best model to {ckpt_path} (val_loss: {val_loss_avg:.4f})", flush=True)

    print(f"Training completed! Best validation loss: {best_val_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
