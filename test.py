# test.py
# Evaluate low-MIMO radar heatmap enhancement.
#
# Compares:
#   baseline = input low-MIMO heatmap, e.g., 1x4
#   model    = predicted enhanced heatmap
# against:
#   target   = 1x8 teacher heatmap
#
# Metrics:
#   Image / heatmap restoration:
#       MAE, MSE, RMSE, PSNR, SSIM
#
#   Sensing-oriented:
#       peak localization error
#       center-of-mass localization error
#       thresholded IoU
#       top-k overlap
#       object-region MAE/MSE

from pathlib import Path
import argparse
import sys
import json
import csv
import math
import time

print("Loaded test.py", flush=True)

import torch
import torch.nn.functional as F

# Do not import matplotlib.pyplot here.
# It can hang on headless servers.
import matplotlib
matplotlib.use("Agg")


PROJECT_ROOT = Path("/mnt/ssd-nas/foysal/Sparse")


# ============================================================
# Basic setup
# ============================================================

def setup_project_root(project_root: Path):
    project_root = Path(project_root)

    if not project_root.exists():
        raise FileNotFoundError(f"Project root does not exist: {project_root}")

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return project_root


def psnr_from_mse(mse: float, max_val: float = 1.0):
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def get_ckpt_arg(ckpt_args, name, default):
    """
    Support both dict and argparse.Namespace saved in checkpoint.
    """
    if isinstance(ckpt_args, dict):
        return ckpt_args.get(name, default)

    return getattr(ckpt_args, name, default)


# ============================================================
# Image restoration metrics
# ============================================================

def ssim_batch(pred, target, window_size=7, c1=0.01 ** 2, c2=0.03 ** 2):
    """
    Windowed SSIM for normalized heatmaps in [0, 1].

    Args:
        pred, target: [B, 1, H, W]

    Returns:
        ssim_per_sample: [B]
    """
    pad = window_size // 2

    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=pad)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.avg_pool2d(pred * pred, window_size, stride=1, padding=pad) - mu_x2
    sigma_y2 = F.avg_pool2d(target * target, window_size, stride=1, padding=pad) - mu_y2
    sigma_xy = F.avg_pool2d(pred * target, window_size, stride=1, padding=pad) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-8

    ssim_map = numerator / denominator

    return ssim_map.mean(dim=(1, 2, 3))


def image_metrics_batch(pred, target):
    """
    Args:
        pred, target: [B, 1, H, W]

    Returns:
        dict of per-sample tensors [B]
    """
    diff = pred - target

    mae = torch.mean(torch.abs(diff), dim=(1, 2, 3))
    mse = torch.mean(diff ** 2, dim=(1, 2, 3))
    rmse = torch.sqrt(mse + 1e-12)

    psnr = torch.tensor(
        [psnr_from_mse(float(v.item())) for v in mse],
        device=pred.device,
        dtype=pred.dtype,
    )

    ssim = ssim_batch(pred, target)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "psnr": psnr,
        "ssim": ssim,
    }


# ============================================================
# Sensing-oriented metrics
# ============================================================

def get_peak_coords(x):
    """
    Peak coordinate for each heatmap.

    Args:
        x: [B, 1, H, W]

    Returns:
        range_idx: [B]
        angle_idx: [B]
    """
    b, _, h, w = x.shape
    flat = x.reshape(b, -1)
    idx = torch.argmax(flat, dim=1)

    range_idx = idx // w
    angle_idx = idx % w

    return range_idx.float(), angle_idx.float()


def get_center_of_mass(x, eps=1e-8):
    """
    Center of mass for each heatmap.

    Args:
        x: [B, 1, H, W]

    Returns:
        range_com: [B]
        angle_com: [B]
    """
    b, _, h, w = x.shape

    x = torch.clamp(x, min=0.0)

    device = x.device
    dtype = x.dtype

    range_grid = torch.arange(h, device=device, dtype=dtype).view(1, 1, h, 1)
    angle_grid = torch.arange(w, device=device, dtype=dtype).view(1, 1, 1, w)

    mass = torch.sum(x, dim=(1, 2, 3)) + eps

    range_com = torch.sum(x * range_grid, dim=(1, 2, 3)) / mass
    angle_com = torch.sum(x * angle_grid, dim=(1, 2, 3)) / mass

    return range_com, angle_com


def make_threshold_mask(x, method="percentile", percentile=95.0, threshold=0.5):
    """
    Make binary heatmap mask.

    method:
        percentile:
            threshold each sample using its own percentile.

        fixed:
            use fixed threshold.
    """
    b, _, h, w = x.shape

    if method == "percentile":
        flat = x.reshape(b, -1)
        q = torch.quantile(flat, percentile / 100.0, dim=1)
        q = q.view(b, 1, 1, 1)
        mask = x >= q

    elif method == "fixed":
        mask = x >= threshold

    else:
        raise ValueError("method must be 'percentile' or 'fixed'.")

    return mask


def iou_batch(pred_mask, target_mask, eps=1e-8):
    """
    IoU per sample.
    """
    pred_mask = pred_mask.bool()
    target_mask = target_mask.bool()

    inter = torch.logical_and(pred_mask, target_mask).float().sum(dim=(1, 2, 3))
    union = torch.logical_or(pred_mask, target_mask).float().sum(dim=(1, 2, 3))

    return inter / (union + eps)


def topk_overlap_batch(pred, target, topk_percent=5.0):
    """
    Top-k overlap.

    For each sample:
        select top-k pixels from pred and target.
        return intersection / k.
    """
    b, _, h, w = pred.shape
    n_pixels = h * w
    k = max(1, int((topk_percent / 100.0) * n_pixels))

    pred_flat = pred.reshape(b, -1)
    target_flat = target.reshape(b, -1)

    overlaps = []

    for i in range(b):
        pred_idx = torch.topk(pred_flat[i], k=k, largest=True).indices
        target_idx = torch.topk(target_flat[i], k=k, largest=True).indices

        pred_set = set(pred_idx.detach().cpu().tolist())
        target_set = set(target_idx.detach().cpu().tolist())

        inter = len(pred_set.intersection(target_set))
        overlaps.append(inter / k)

    return torch.tensor(overlaps, device=pred.device, dtype=pred.dtype)


def object_region_error_batch(pred, target, target_mask, eps=1e-8):
    """
    MAE/MSE only inside target object region.
    """
    mask = target_mask.float()

    diff = pred - target

    denom = mask.sum(dim=(1, 2, 3)) + eps

    obj_mae = (torch.abs(diff) * mask).sum(dim=(1, 2, 3)) / denom
    obj_mse = ((diff ** 2) * mask).sum(dim=(1, 2, 3)) / denom

    return obj_mae, obj_mse


def sensing_metrics_batch(
    pred,
    target,
    mask_method="percentile",
    mask_percentile=95.0,
    fixed_threshold=0.5,
    topk_percent=5.0,
):
    """
    Sensing-oriented metrics per sample.
    """
    # Peak localization
    pred_r_peak, pred_a_peak = get_peak_coords(pred)
    target_r_peak, target_a_peak = get_peak_coords(target)

    peak_range_error = torch.abs(pred_r_peak - target_r_peak)
    peak_angle_error = torch.abs(pred_a_peak - target_a_peak)

    peak_l2_error = torch.sqrt(
        peak_range_error ** 2 + peak_angle_error ** 2
    )

    # Center-of-mass localization
    pred_r_com, pred_a_com = get_center_of_mass(pred)
    target_r_com, target_a_com = get_center_of_mass(target)

    com_range_error = torch.abs(pred_r_com - target_r_com)
    com_angle_error = torch.abs(pred_a_com - target_a_com)

    com_l2_error = torch.sqrt(
        com_range_error ** 2 + com_angle_error ** 2
    )

    # Masks and overlap
    pred_mask = make_threshold_mask(
        pred,
        method=mask_method,
        percentile=mask_percentile,
        threshold=fixed_threshold,
    )

    target_mask = make_threshold_mask(
        target,
        method=mask_method,
        percentile=mask_percentile,
        threshold=fixed_threshold,
    )

    iou = iou_batch(pred_mask, target_mask)

    topk_overlap = topk_overlap_batch(
        pred,
        target,
        topk_percent=topk_percent,
    )

    obj_mae, obj_mse = object_region_error_batch(
        pred,
        target,
        target_mask,
    )

    return {
        "peak_range_error": peak_range_error,
        "peak_angle_error": peak_angle_error,
        "peak_l2_error": peak_l2_error,
        "com_range_error": com_range_error,
        "com_angle_error": com_angle_error,
        "com_l2_error": com_l2_error,
        "iou": iou,
        "topk_overlap": topk_overlap,
        "object_mae": obj_mae,
        "object_mse": obj_mse,
    }


def compute_all_metrics(
    pred,
    target,
    mask_method="percentile",
    mask_percentile=95.0,
    fixed_threshold=0.5,
    topk_percent=5.0,
):
    image_metrics = image_metrics_batch(pred, target)

    sensing_metrics = sensing_metrics_batch(
        pred,
        target,
        mask_method=mask_method,
        mask_percentile=mask_percentile,
        fixed_threshold=fixed_threshold,
        topk_percent=topk_percent,
    )

    out = {}
    out.update(image_metrics)
    out.update(sensing_metrics)

    return out


# ============================================================
# Aggregation and saving
# ============================================================

def init_metric_storage(prefix):
    keys = [
        "mae",
        "mse",
        "rmse",
        "psnr",
        "ssim",
        "peak_range_error",
        "peak_angle_error",
        "peak_l2_error",
        "com_range_error",
        "com_angle_error",
        "com_l2_error",
        "iou",
        "topk_overlap",
        "object_mae",
        "object_mse",
    ]

    return {f"{prefix}_{k}": [] for k in keys}


def append_metrics(storage, prefix, metrics):
    for k, v in metrics.items():
        key = f"{prefix}_{k}"
        storage[key].extend(v.detach().cpu().tolist())


def summarize_metric_storage(storage):
    summary = {}

    for k, values in storage.items():
        if len(values) == 0:
            continue

        t = torch.tensor(values, dtype=torch.float32)

        summary[k] = {
            "mean": float(t.mean().item()),
            "std": float(t.std(unbiased=False).item()),
            "median": float(t.median().item()),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
        }

    return summary


def save_summary_json(summary, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)


def save_per_sample_csv(rows, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Visualization
# ============================================================

@torch.no_grad()
def save_sample_figures(
    model,
    loader,
    device,
    save_dir,
    max_batches=1,
    max_items_per_batch=4,
):
    import matplotlib.pyplot as plt

    model.eval()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        print(f"Saving figure batch {batch_idx + 1}", flush=True)

        x = batch["input"].to(device)
        y = batch["target"].to(device)
        pred = model(x)

        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        pred_cpu = pred.detach().cpu()

        n = min(max_items_per_batch, x_cpu.shape[0])

        for i in range(n):
            sample_id = str(batch["sample_id"][i])

            inp = x_cpu[i, 0].numpy()
            out = pred_cpu[i, 0].numpy()
            tgt = y_cpu[i, 0].numpy()

            err_inp = abs(inp - tgt)
            err_pred = abs(out - tgt)

            fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))

            items = [
                (inp, "Input low-MIMO"),
                (out, "Predicted"),
                (tgt, "Target 1x8"),
                (err_inp, "|Input - Target|"),
                (err_pred, "|Pred - Target|"),
            ]

            for ax, (img, title) in zip(axes, items):
                im = ax.imshow(
                    img,
                    aspect="auto",
                    origin="lower",
                    vmin=0.0,
                    vmax=1.0,
                )
                ax.set_title(title)
                ax.set_xlabel("Angle bin")
                ax.set_ylabel("Range bin")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.suptitle(sample_id)
            plt.tight_layout()

            safe_sample_id = sample_id.replace("/", "_")
            save_path = save_dir / f"{count:04d}_{safe_sample_id}.png"
            plt.savefig(save_path, dpi=200)
            plt.close()

            count += 1


# ============================================================
# Main evaluation
# ============================================================

@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
    args,
):
    print("Starting evaluation loop...", flush=True)

    model.eval()

    storage = {}
    storage.update(init_metric_storage("baseline"))
    storage.update(init_metric_storage("model"))

    per_sample_rows = []

    start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        print(f"Processing batch {batch_idx + 1}/{len(loader)}", flush=True)

        x = batch["input"].to(device, non_blocking=False)
        y = batch["target"].to(device, non_blocking=False)

        pred = model(x)

        baseline_metrics = compute_all_metrics(
            pred=x,
            target=y,
            mask_method=args.mask_method,
            mask_percentile=args.mask_percentile,
            fixed_threshold=args.fixed_threshold,
            topk_percent=args.topk_percent,
        )

        model_metrics = compute_all_metrics(
            pred=pred,
            target=y,
            mask_method=args.mask_method,
            mask_percentile=args.mask_percentile,
            fixed_threshold=args.fixed_threshold,
            topk_percent=args.topk_percent,
        )

        append_metrics(storage, "baseline", baseline_metrics)
        append_metrics(storage, "model", model_metrics)

        batch_size = x.shape[0]

        for i in range(batch_size):
            row = {
                "sample_id": str(batch["sample_id"][i]),
            }

            for k, v in baseline_metrics.items():
                row[f"baseline_{k}"] = float(v[i].detach().cpu().item())

            for k, v in model_metrics.items():
                row[f"model_{k}"] = float(v[i].detach().cpu().item())

            per_sample_rows.append(row)

        if args.max_test_batches is not None:
            if (batch_idx + 1) >= args.max_test_batches:
                print("Reached max_test_batches. Stopping early.", flush=True)
                break

    elapsed = time.time() - start_time
    print(f"Evaluation loop finished in {elapsed:.2f} seconds.", flush=True)

    summary = summarize_metric_storage(storage)

    return summary, per_sample_rows


def print_key_results(summary):
    def m(key):
        return summary[key]["mean"]

    print("\n========================================", flush=True)
    print("Key test results", flush=True)
    print("========================================", flush=True)

    print("\nImage / heatmap restoration metrics", flush=True)
    print("-----------------------------------", flush=True)
    print(f"Baseline PSNR : {m('baseline_psnr'):.2f} dB", flush=True)
    print(f"Model PSNR    : {m('model_psnr'):.2f} dB", flush=True)
    print(f"PSNR gain     : {m('model_psnr') - m('baseline_psnr'):.2f} dB", flush=True)
    print(f"Baseline SSIM : {m('baseline_ssim'):.4f}", flush=True)
    print(f"Model SSIM    : {m('model_ssim'):.4f}", flush=True)
    print(f"Baseline MAE  : {m('baseline_mae'):.6f}", flush=True)
    print(f"Model MAE     : {m('model_mae'):.6f}", flush=True)
    print(f"Baseline MSE  : {m('baseline_mse'):.6f}", flush=True)
    print(f"Model MSE     : {m('model_mse'):.6f}", flush=True)

    print("\nSensing-oriented metrics", flush=True)
    print("------------------------", flush=True)
    print(f"Baseline peak L2 error : {m('baseline_peak_l2_error'):.3f} bins", flush=True)
    print(f"Model peak L2 error    : {m('model_peak_l2_error'):.3f} bins", flush=True)
    print(f"Baseline COM L2 error  : {m('baseline_com_l2_error'):.3f} bins", flush=True)
    print(f"Model COM L2 error     : {m('model_com_l2_error'):.3f} bins", flush=True)
    print(f"Baseline IoU           : {m('baseline_iou'):.4f}", flush=True)
    print(f"Model IoU              : {m('model_iou'):.4f}", flush=True)
    print(f"Baseline top-k overlap : {m('baseline_topk_overlap'):.4f}", flush=True)
    print(f"Model top-k overlap    : {m('model_topk_overlap'):.4f}", flush=True)
    print(f"Baseline object MAE    : {m('baseline_object_mae'):.6f}", flush=True)
    print(f"Model object MAE       : {m('model_object_mae'):.6f}", flush=True)


def main():
    print("Started main()", flush=True)

    parser = argparse.ArgumentParser(
        description="Evaluate radar heatmap model with restoration and sensing metrics."
    )

    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT))

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/radar_heatmap_training/checkpoints/best.pt",
    )

    parser.add_argument("--input_config", type=str, default="1x4")
    parser.add_argument("--target_config", type=str, default="1x8")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--normalize_mode", type=str, default="fixed_db")
    parser.add_argument("--db_min", type=float, default=30.0)
    parser.add_argument("--db_max", type=float, default=75.0)

    parser.add_argument(
        "--mask_method",
        type=str,
        default="percentile",
        choices=["percentile", "fixed"],
    )

    parser.add_argument(
        "--mask_percentile",
        type=float,
        default=95.0,
        help="Used when mask_method=percentile.",
    )

    parser.add_argument(
        "--fixed_threshold",
        type=float,
        default=0.5,
        help="Used when mask_method=fixed.",
    )

    parser.add_argument(
        "--topk_percent",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/radar_heatmap_eval",
    )

    parser.add_argument(
        "--save_figures",
        action="store_true",
    )

    parser.add_argument(
        "--num_figure_batches",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max_test_batches",
        type=int,
        default=None,
        help="Use this for debugging. Example: --max_test_batches 2",
    )

    args = parser.parse_args()
    print("Parsed arguments.", flush=True)

    project_root = setup_project_root(Path(args.project_root))
    print(f"Project root checked: {project_root}", flush=True)

    print("Importing heatmap_dataloader and model...", flush=True)

    from heatmap_dataloader import create_radar_heatmap_dataloaders
    from model_residual_attention import RadarHeatmapUNet

    print("Imported heatmap_dataloader and model.", flush=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path

    print(f"Checking checkpoint: {checkpoint_path}", flush=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("\n========================================", flush=True)
    print("Radar heatmap evaluation", flush=True)
    print("========================================", flush=True)
    print(f"Project root : {project_root}", flush=True)
    print(f"Checkpoint   : {checkpoint_path}", flush=True)
    print(f"Input config : {args.input_config}", flush=True)
    print(f"Target config: {args.target_config}", flush=True)
    print(f"Device       : {device}", flush=True)

    print("Creating dataloaders...", flush=True)

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

    print("Dataloaders created.", flush=True)
    print(f"Train batches: {len(train_loader)}", flush=True)
    print(f"Val batches  : {len(val_loader)}", flush=True)
    print(f"Test batches : {len(test_loader)}", flush=True)

    print("Loading checkpoint...", flush=True)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    ckpt_args = checkpoint.get("args", {})
    base_channels = get_ckpt_arg(ckpt_args, "base_channels", args.base_channels)

    print(f"Using base_channels={base_channels}", flush=True)

    print("Creating model...", flush=True)

    model = RadarHeatmapUNet(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
        use_sigmoid=False,
    ).to(device)

    print("Loading model weights...", flush=True)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    print("Model loaded.", flush=True)

    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, per_sample_rows = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        args=args,
    )

    print_key_results(summary)

    summary_path = out_dir / f"summary_{args.input_config}_to_{args.target_config}.json"
    csv_path = out_dir / f"per_sample_{args.input_config}_to_{args.target_config}.csv"

    save_summary_json(summary, summary_path)
    save_per_sample_csv(per_sample_rows, csv_path)

    print("\nSaved:", flush=True)
    print(f"  {summary_path}", flush=True)
    print(f"  {csv_path}", flush=True)

    if args.save_figures:
        fig_dir = out_dir / f"figures_{args.input_config}_to_{args.target_config}"

        print("Saving sample figures...", flush=True)

        save_sample_figures(
            model=model,
            loader=test_loader,
            device=device,
            save_dir=fig_dir,
            max_batches=args.num_figure_batches,
            max_items_per_batch=4,
        )

        print(f"  {fig_dir}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()