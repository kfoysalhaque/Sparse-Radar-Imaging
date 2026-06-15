# train.py
# Train sparse MIMO FMCW radar image reconstruction model.

from pathlib import Path
import csv
import time

import torch
import torch.nn as nn
import torch.optim as optim

import config
from dataloader import create_train_val_loaders
from model import SparseRadarReconNet, SimpleCNNBaseline, count_parameters


# ============================================================
# Utility functions
# ============================================================

def get_device():
    """
    Select CUDA if available, otherwise CPU.
    """
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def nmse_loss(pred, target, eps=1e-8):
    """
    Normalized mean squared error.
    """
    mse = torch.sum((pred - target) ** 2, dim=(1, 2, 3))
    denom = torch.sum(target ** 2, dim=(1, 2, 3)) + eps
    nmse = mse / denom
    return torch.mean(nmse)


def train_one_epoch(
    model,
    loader,
    optimizer,
    l1_criterion,
    device,
    use_mask=True,
):
    """
    Train for one epoch.
    """
    model.train()

    total_loss = 0.0
    total_l1 = 0.0
    total_nmse = 0.0
    num_batches = 0

    for batch in loader:
        sparse = batch["sparse"].to(device)  # [B, 1, 128, 64]
        full = batch["full"].to(device)      # [B, 1, 128, 64]
        mask = batch["mask"].to(device)      # [B, 8]

        optimizer.zero_grad()

        if use_mask:
            pred = model(sparse, mask)
        else:
            pred = model(sparse)

        l1 = l1_criterion(pred, full)
        nmse = nmse_loss(pred, full)

        # Main training loss
        loss = l1 + 0.1 * nmse

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_l1 += l1.item()
        total_nmse += nmse.item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "l1": total_l1 / num_batches,
        "nmse": total_nmse / num_batches,
    }


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    l1_criterion,
    device,
    use_mask=True,
):
    """
    Validate for one epoch.
    """
    model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_nmse = 0.0
    num_batches = 0

    for batch in loader:
        sparse = batch["sparse"].to(device)
        full = batch["full"].to(device)
        mask = batch["mask"].to(device)

        if use_mask:
            pred = model(sparse, mask)
        else:
            pred = model(sparse)

        l1 = l1_criterion(pred, full)
        nmse = nmse_loss(pred, full)
        loss = l1 + 0.1 * nmse

        total_loss += loss.item()
        total_l1 += l1.item()
        total_nmse += nmse.item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "l1": total_l1 / num_batches,
        "nmse": total_nmse / num_batches,
    }


def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    save_path,
):
    """
    Save model checkpoint.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        save_path,
    )


def write_log_header(log_path):
    """
    Create CSV log file with header.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_l1",
                "train_nmse",
                "val_loss",
                "val_l1",
                "val_nmse",
                "epoch_time_sec",
            ]
        )


def append_log(log_path, row):
    """
    Append one row to CSV log file.
    """
    with open(log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# ============================================================
# Main training function
# ============================================================

def main():
    config.create_dirs()

    device = get_device()
    print(f"Using device: {device}")

    # ------------------------------------------------------------
    # Choose model
    # ------------------------------------------------------------
    # Proposed model:
    model_name = "sparse_radar_recon"
    model = SparseRadarReconNet()
    use_mask = True

    # Baseline model:
    # model_name = "simple_cnn_baseline"
    # model = SimpleCNNBaseline()
    # use_mask = False

    model = model.to(device)

    print(f"Model: {model_name}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    # ------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------
    train_loader, val_loader = create_train_val_loaders(
        ratio_filter=None
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # ------------------------------------------------------------
    # Loss and optimizer
    # ------------------------------------------------------------
    l1_criterion = nn.L1Loss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    # ------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------
    checkpoint_path = config.CHECKPOINT_DIR / f"{model_name}_best.pth"
    log_path = config.LOG_DIR / f"{model_name}_train_log.csv"

    write_log_header(log_path)

    best_val_loss = float("inf")

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    for epoch in range(1, config.NUM_EPOCHS + 1):
        start_time = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            l1_criterion=l1_criterion,
            device=device,
            use_mask=use_mask,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            l1_criterion=l1_criterion,
            device=device,
            use_mask=use_mask,
        )

        scheduler.step(val_metrics["loss"])

        epoch_time = time.time() - start_time

        print(
            f"Epoch [{epoch:03d}/{config.NUM_EPOCHS}] "
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Train L1: {train_metrics['l1']:.6f} | "
            f"Train NMSE: {train_metrics['nmse']:.6f} || "
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"Val L1: {val_metrics['l1']:.6f} | "
            f"Val NMSE: {val_metrics['nmse']:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        append_log(
            log_path,
            [
                epoch,
                train_metrics["loss"],
                train_metrics["l1"],
                train_metrics["nmse"],
                val_metrics["loss"],
                val_metrics["l1"],
                val_metrics["nmse"],
                epoch_time,
            ],
        )

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=best_val_loss,
                save_path=checkpoint_path,
            )

            print(f"  Saved best model: {checkpoint_path}")

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
