# test.py
# Evaluate sparse MIMO FMCW radar image reconstruction models.

from pathlib import Path
from collections import defaultdict
import argparse
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt

import config
from dataloader import create_dataloader
from model import SparseRadarReconNet, SimpleCNNBaseline


# ============================================================
# Device and model
# ============================================================

def get_device():
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(model_name: str):
    if model_name == "sparse_radar_recon":
        model = SparseRadarReconNet()
        use_mask = True
    elif model_name == "simple_cnn_baseline":
        model = SimpleCNNBaseline()
        use_mask = False
    else:
        raise ValueError(
            "model_name must be 'sparse_radar_recon' or 'simple_cnn_baseline'"
        )

    return model, use_mask


def load_checkpoint(model, checkpoint_path: Path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Checkpoint val loss: {ckpt.get('val_loss', 'unknown')}")

    return model


# ============================================================
# Metrics
# ============================================================

def batch_global_ssim(pred, target, eps=1e-8):
    """
    Simple global SSIM per image.

    This is enough for debugging and comparison.
    Later, for paper-quality evaluation, we can also use skimage SSIM.
    """
    x = pred.flatten(start_dim=1)
    y = target.flatten(start_dim=1)

    mu_x = x.mean(dim=1)
    mu_y = y.mean(dim=1)

    var_x = ((x - mu_x[:, None]) ** 2).mean(dim=1)
    var_y = ((y - mu_y[:, None]) ** 2).mean(dim=1)
    cov_xy = ((x - mu_x[:, None]) * (y - mu_y[:, None])).mean(dim=1)

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2) + eps
    )

    return ssim


def compute_metrics(pred, target, eps=1e-8):
    """
    Compute per-sample metrics.

    Inputs:
        pred:   [B, 1, H, W]
        target: [B, 1, H, W]
    """
    diff = pred - target

    l1 = torch.mean(torch.abs(diff), dim=(1, 2, 3))
    mse = torch.mean(diff ** 2, dim=(1, 2, 3))

    nmse = torch.sum(diff ** 2, dim=(1, 2, 3)) / (
        torch.sum(target ** 2, dim=(1, 2, 3)) + eps
    )

    psnr = 10.0 * torch.log10(1.0 / (mse + eps))
    ssim = batch_global_ssim(pred, target, eps=eps)

    return {
        "l1": l1,
        "mse": mse,
        "nmse": nmse,
        "psnr": psnr,
        "ssim": ssim,
    }


# ============================================================
# Visualization
# ============================================================

def save_example_figure(
    sparse_img,
    pred_img,
    full_img,
    save_path: Path,
    title: str,
):
    """
    Save one visual comparison:
        sparse input, reconstructed output, full target,
        input error, output error.
    """
    sparse_img = sparse_img.squeeze()
    pred_img = pred_img.squeeze()
    full_img = full_img.squeeze()

    err_sparse = np.abs(sparse_img - full_img)
    err_pred = np.abs(pred_img - full_img)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(16, 3.5))

    images = [
        sparse_img,
        pred_img,
        full_img,
        err_sparse,
        err_pred,
    ]

    titles = [
        "Sparse Input",
        "Reconstructed",
        "Full Target",
        "|Sparse - Full|",
        "|Recon - Full|",
    ]

    for i, (img, t) in enumerate(zip(images, titles)):
        plt.subplot(1, 5, i + 1)

        if i < 3:
            plt.imshow(img, aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
        else:
            plt.imshow(img, aspect="auto", origin="lower", vmin=0.0, vmax=max(err_sparse.max(), err_pred.max()))

        plt.title(t)
        plt.xlabel("Angle bin")
        if i == 0:
            plt.ylabel("Range bin")
        plt.colorbar(fraction=0.046, pad=0.04)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    use_mask: bool,
    model_name: str,
    num_examples: int,
):
    model.eval()

    results_dir = config.RESULTS_DIR / model_name
    fig_dir = config.FIGURE_DIR / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    sample_csv_path = results_dir / "test_sample_metrics.csv"
    summary_csv_path = results_dir / "test_summary_metrics.csv"
    ratio_csv_path = results_dir / "test_ratio_metrics.csv"

    fieldnames = [
        "sequence",
        "frame_id",
        "ratio",
        "sparse_l1",
        "sparse_mse",
        "sparse_nmse",
        "sparse_psnr",
        "sparse_ssim",
        "pred_l1",
        "pred_mse",
        "pred_nmse",
        "pred_psnr",
        "pred_ssim",
    ]

    total = 0
    global_sums = defaultdict(float)
    ratio_sums = defaultdict(lambda: defaultdict(float))
    ratio_counts = defaultdict(int)

    example_count = 0

    with open(sample_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for batch_idx, batch in enumerate(loader):
            sparse = batch["sparse"].to(device)
            full = batch["full"].to(device)
            mask = batch["mask"].to(device)
            ratios = batch["ratio"].cpu().numpy()

            if use_mask:
                pred = model(sparse, mask)
            else:
                pred = model(sparse)

            sparse_metrics = compute_metrics(sparse, full)
            pred_metrics = compute_metrics(pred, full)

            batch_size = sparse.shape[0]
            total += batch_size

            for key in sparse_metrics:
                global_sums[f"sparse_{key}"] += sparse_metrics[key].sum().item()
                global_sums[f"pred_{key}"] += pred_metrics[key].sum().item()

            sequences = batch["sequence"]
            frame_ids = batch["frame_id"]

            sparse_cpu = sparse.cpu().numpy()
            pred_cpu = pred.cpu().numpy()
            full_cpu = full.cpu().numpy()

            for i in range(batch_size):
                ratio = float(ratios[i])
                ratio_key = round(ratio, 2)

                row = {
                    "sequence": sequences[i],
                    "frame_id": frame_ids[i],
                    "ratio": ratio,
                }

                for key in sparse_metrics:
                    row[f"sparse_{key}"] = sparse_metrics[key][i].item()
                    row[f"pred_{key}"] = pred_metrics[key][i].item()

                    ratio_sums[ratio_key][f"sparse_{key}"] += sparse_metrics[key][i].item()
                    ratio_sums[ratio_key][f"pred_{key}"] += pred_metrics[key][i].item()

                ratio_counts[ratio_key] += 1

                writer.writerow(row)

                if example_count < num_examples:
                    save_path = fig_dir / f"example_{example_count:03d}_{sequences[i]}_{frame_ids[i]}_r{int(ratio * 100):03d}.png"

                    title = f"{model_name} | {sequences[i]} | frame {frame_ids[i]} | ratio {ratio:.2f}"

                    save_example_figure(
                        sparse_img=sparse_cpu[i],
                        pred_img=pred_cpu[i],
                        full_img=full_cpu[i],
                        save_path=save_path,
                        title=title,
                    )

                    example_count += 1

            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {batch_idx + 1}/{len(loader)} batches")

    # Global averages
    global_avg = {key: value / total for key, value in global_sums.items()}

    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])

        for key, value in global_avg.items():
            writer.writerow([key, value])

    # Ratio-wise averages
    with open(ratio_csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        header = ["ratio", "count"] + sorted(next(iter(ratio_sums.values())).keys())
        writer.writerow(header)

        for ratio_key in sorted(ratio_sums.keys()):
            count = ratio_counts[ratio_key]
            row = [ratio_key, count]

            for metric_name in sorted(ratio_sums[ratio_key].keys()):
                row.append(ratio_sums[ratio_key][metric_name] / count)

            writer.writerow(row)

    print("\nEvaluation complete.")
    print(f"Total test samples: {total}")
    print(f"Saved sample metrics: {sample_csv_path}")
    print(f"Saved summary metrics: {summary_csv_path}")
    print(f"Saved ratio-wise metrics: {ratio_csv_path}")
    print(f"Saved example figures: {fig_dir}")

    print("\nGlobal average metrics:")
    for key, value in global_avg.items():
        print(f"  {key}: {value:.6f}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test sparse MIMO FMCW radar image reconstruction model."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="sparse_radar_recon",
        choices=["sparse_radar_recon", "simple_cnn_baseline"],
        help="Model to evaluate.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. If not provided, uses checkpoints/{model}_best.pth.",
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="Optional ratio filter, e.g., 0.5 to test only r050 samples.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size for testing.",
    )

    parser.add_argument(
        "--num_examples",
        type=int,
        default=20,
        help="Number of visual examples to save.",
    )

    args = parser.parse_args()

    config.create_dirs()

    device = get_device()
    print(f"Using device: {device}")

    model, use_mask = build_model(args.model)
    model = model.to(device)

    if args.checkpoint is None:
        checkpoint_path = config.CHECKPOINT_DIR / f"{args.model}_best.pth"
    else:
        checkpoint_path = Path(args.checkpoint)

    model = load_checkpoint(model, checkpoint_path, device)

    test_loader = create_dataloader(
        split="test",
        batch_size=args.batch_size,
        shuffle=False,
        ratio_filter=args.ratio,
    )

    print(f"Test batches: {len(test_loader)}")

    evaluate(
        model=model,
        loader=test_loader,
        device=device,
        use_mask=use_mask,
        model_name=args.model,
        num_examples=args.num_examples,
    )


if __name__ == "__main__":
    main()