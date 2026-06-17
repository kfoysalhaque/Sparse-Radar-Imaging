# train_radar_heatmap.py
# Train low-MIMO radar heatmap enhancement:
# 1x4 / 1x2 / 1x6 radar heatmap -> 1x8 teacher radar heatmap

from pathlib import Path
import argparse
import sys
import time
import math

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/media/foysal/Foysal-2/Github/Sparse-Radar-Imaging")


def setup_project_root(project_root: Path):
    project_root = Path(project_root)

    if not project_root.exists():
        raise FileNotFoundError(f"Project root does not exist: {project_root}")

    sys.path.insert(0, str(project_root))
    return project_root


def psnr_from_mse(mse: float, max_val: float = 1.0):
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def save_checkpoint(
    save_path,
    model,
    optimizer,
    epoch,
    best_val_loss,
    args,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args),
        },
        save_path,
    )


@torch.no_grad()
def save_prediction_figure(
    model,
    batch,
    device,
    save_path,
    max_items=3,
):
    model.eval()

    x = batch["input"].to(device)
    y = batch["target"].to(device)

    pred = model(x)

    x = x.detach().cpu()
    y = y.detach().cpu()
    pred = pred.detach().cpu()

    n = min(max_items, x.shape[0])

    fig, axes = plt.subplots(
        n,
        3,
        figsize=(10, 3.2 * n),
    )

    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        sample_id = batch["sample_id"][i]

        imgs = [
            x[i, 0].numpy(),
            pred[i, 0].numpy(),
            y[i, 0].numpy(),
        ]

        titles = [
            f"Input 1x4\n{sample_id}",
            "Predicted 1x8-like",
            "Target 1x8",
        ]

        for j in range(3):
            ax = axes[i, j]
            im = ax.imshow(
                imgs[j],
                aspect="auto",
                origin="lower",
                vmin=0.0,
                vmax=1.0,
            )
            ax.set_title(titles[j])
            ax.set_xlabel("Angle bin")
            ax.set_ylabel("Range bin")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=200)
    plt.close()


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler=None,
    use_amp=False,
):
    model.train()

    total_loss = 0.0
    total_l1 = 0.0
    total_mse = 0.0
    total_samples = 0

    l1_metric = nn.L1Loss(reduction="sum")
    mse_metric = nn.MSELoss(reduction="sum")

    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast():
                pred = model(x)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        bsz = x.shape[0]

        total_loss += loss.item() * bsz
        total_l1 += l1_metric(pred, y).item()
        total_mse += mse_metric(pred, y).item()
        total_samples += bsz

    num_pixels = total_samples * y.shape[1] * y.shape[2] * y.shape[3]

    avg_loss = total_loss / total_samples
    avg_l1 = total_l1 / num_pixels
    avg_mse = total_mse / num_pixels
    avg_psnr = psnr_from_mse(avg_mse)

    return {
        "loss": avg_loss,
        "l1": avg_l1,
        "mse": avg_mse,
        "psnr": avg_psnr,
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_mse = 0.0
    total_samples = 0

    l1_metric = nn.L1Loss(reduction="sum")
    mse_metric = nn.MSELoss(reduction="sum")

    first_batch = None

    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        pred = model(x)
        loss = criterion(pred, y)

        bsz = x.shape[0]

        total_loss += loss.item() * bsz
        total_l1 += l1_metric(pred, y).item()
        total_mse += mse_metric(pred, y).item()
        total_samples += bsz

        if first_batch is None:
            first_batch = batch

    num_pixels = total_samples * y.shape[1] * y.shape[2] * y.shape[3]

    avg_loss = total_loss / total_samples
    avg_l1 = total_l1 / num_pixels
    avg_mse = total_mse / num_pixels
    avg_psnr = psnr_from_mse(avg_mse)

    return {
        "loss": avg_loss,
        "l1": avg_l1,
        "mse": avg_mse,
        "psnr": avg_psnr,
        "first_batch": first_batch,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train radar heatmap enhancement model."
    )

    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT))

    parser.add_argument("--input_config", type=str, default="1x4")
    parser.add_argument("--target_config", type=str, default="1x8")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--base_channels", type=int, default=32)

    parser.add_argument("--normalize_mode", type=str, default="fixed_db")
    parser.add_argument("--db_min", type=float, default=30.0)
    parser.add_argument("--db_max", type=float, default=75.0)

    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--mse_weight", type=float, default=0.5)
    parser.add_argument("--grad_weight", type=float, default=0.2)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/radar_heatmap_training",
    )

    parser.add_argument("--save_every", type=int, default=5)

    args = parser.parse_args()

    project_root = setup_project_root(Path(args.project_root))

    from heatmap_dataloader import create_radar_heatmap_dataloaders
    from model import RadarHeatmapUNet, RadarHeatmapLoss, count_parameters

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    out_dir = project_root / args.out_dir
    ckpt_dir = out_dir / "checkpoints"
    fig_dir = out_dir / "figures"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("\n========================================")
    print("Radar heatmap training")
    print("========================================")
    print(f"Project root : {project_root}")
    print(f"Input config : {args.input_config}")
    print(f"Target config: {args.target_config}")
    print(f"Device       : {device}")
    print(f"Output dir   : {out_dir}")

    train_loader, val_loader, test_loader = create_radar_heatmap_dataloaders(
        project_root=project_root,
        input_config=args.input_config,
        target_config=args.target_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize_mode=args.normalize_mode,
        db_min=args.db_min,
        db_max=args.db_max,
    )

    model = RadarHeatmapUNet(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        use_sigmoid=True,
    ).to(device)

    criterion = RadarHeatmapLoss(
        l1_weight=args.l1_weight,
        mse_weight=args.mse_weight,
        grad_weight=args.grad_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05,
    )

    scaler = GradScaler(enabled=args.amp)

    print(f"Trainable parameters: {count_parameters(model):,}")

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=args.amp,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        elapsed = time.time() - start_time
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"lr={lr_now:.2e} "
            f"time={elapsed:.1f}s | "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_l1={train_metrics['l1']:.6f} "
            f"train_psnr={train_metrics['psnr']:.2f} | "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_l1={val_metrics['l1']:.6f} "
            f"val_psnr={val_metrics['psnr']:.2f}"
        )

        latest_path = ckpt_dir / "latest.pt"

        save_checkpoint(
            save_path=latest_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            args=args,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            best_path = ckpt_dir / "best.pt"

            save_checkpoint(
                save_path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                args=args,
            )

            print(f"  Saved best checkpoint: {best_path}")

            save_prediction_figure(
                model=model,
                batch=val_metrics["first_batch"],
                device=device,
                save_path=fig_dir / f"best_epoch_{epoch:03d}.png",
                max_items=3,
            )

        if epoch % args.save_every == 0:
            save_prediction_figure(
                model=model,
                batch=val_metrics["first_batch"],
                device=device,
                save_path=fig_dir / f"epoch_{epoch:03d}.png",
                max_items=3,
            )

    print("\nTraining finished.")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Best checkpoint: {ckpt_dir / 'best.pt'}")

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\nTest results")
    print("------------")
    print(f"test_loss={test_metrics['loss']:.6f}")
    print(f"test_l1  ={test_metrics['l1']:.6f}")
    print(f"test_mse ={test_metrics['mse']:.6f}")
    print(f"test_psnr={test_metrics['psnr']:.2f}")


if __name__ == "__main__":
    main()