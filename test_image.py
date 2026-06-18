from pathlib import Path
import argparse
import csv
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")


PROJECT_ROOT = Path("/media/foysal/Foysal-2/Github/Sparse-Radar-Imaging")


def setup_project_root(project_root: Path) -> Path:
    project_root = Path(project_root)

    if not project_root.exists():
        raise FileNotFoundError(f"Project root does not exist: {project_root}")

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return project_root


def psnr_from_mse(mse: float, max_val: float = 1.0) -> float:
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def get_ckpt_arg(ckpt_args, name, default):
    if isinstance(ckpt_args, dict):
        return ckpt_args.get(name, default)
    return getattr(ckpt_args, name, default)


def ssim_batch(pred, target, window_size=7, c1=0.01 ** 2, c2=0.03 ** 2):
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

    return (numerator / denominator).mean(dim=(1, 2, 3))


def image_metrics_batch(pred, target):
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


def get_peak_coords(x):
    b, _, _, w = x.shape
    flat = x.reshape(b, -1)
    idx = torch.argmax(flat, dim=1)
    return (idx // w).float(), (idx % w).float()


def get_center_of_mass(x, eps=1e-8):
    b, _, h, w = x.shape
    x = torch.clamp(x, min=0.0)

    range_grid = torch.arange(h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
    angle_grid = torch.arange(w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)

    mass = torch.sum(x, dim=(1, 2, 3)) + eps
    range_com = torch.sum(x * range_grid, dim=(1, 2, 3)) / mass
    angle_com = torch.sum(x * angle_grid, dim=(1, 2, 3)) / mass

    return range_com, angle_com


def make_threshold_mask(x, method="percentile", percentile=95.0, threshold=0.5):
    b = x.shape[0]

    if method == "percentile":
        flat = x.reshape(b, -1)
        q = torch.quantile(flat, percentile / 100.0, dim=1).view(b, 1, 1, 1)
        return x >= q

    if method == "fixed":
        return x >= threshold

    raise ValueError("method must be 'percentile' or 'fixed'.")


def iou_batch(pred_mask, target_mask, eps=1e-8):
    pred_mask = pred_mask.bool()
    target_mask = target_mask.bool()

    inter = torch.logical_and(pred_mask, target_mask).float().sum(dim=(1, 2, 3))
    union = torch.logical_or(pred_mask, target_mask).float().sum(dim=(1, 2, 3))

    return inter / (union + eps)


def topk_overlap_batch(pred, target, topk_percent=5.0):
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
        overlaps.append(len(pred_set.intersection(target_set)) / k)

    return torch.tensor(overlaps, device=pred.device, dtype=pred.dtype)


def object_region_error_batch(pred, target, target_mask, eps=1e-8):
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
    pred_r_peak, pred_a_peak = get_peak_coords(pred)
    target_r_peak, target_a_peak = get_peak_coords(target)

    peak_range_error = torch.abs(pred_r_peak - target_r_peak)
    peak_angle_error = torch.abs(pred_a_peak - target_a_peak)
    peak_l2_error = torch.sqrt(peak_range_error ** 2 + peak_angle_error ** 2)

    pred_r_com, pred_a_com = get_center_of_mass(pred)
    target_r_com, target_a_com = get_center_of_mass(target)

    com_range_error = torch.abs(pred_r_com - target_r_com)
    com_angle_error = torch.abs(pred_a_com - target_a_com)
    com_l2_error = torch.sqrt(com_range_error ** 2 + com_angle_error ** 2)

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
    topk_overlap = topk_overlap_batch(pred, target, topk_percent=topk_percent)
    obj_mae, obj_mse = object_region_error_batch(pred, target, target_mask)

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
    out = {}
    out.update(image_metrics_batch(pred, target))
    out.update(
        sensing_metrics_batch(
            pred,
            target,
            mask_method=mask_method,
            mask_percentile=mask_percentile,
            fixed_threshold=fixed_threshold,
            topk_percent=topk_percent,
        )
    )
    return out


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
    for key, value in metrics.items():
        storage[f"{prefix}_{key}"].extend(value.detach().cpu().tolist())


def summarize_metric_storage(storage):
    summary = {}

    for key, values in storage.items():
        if not values:
            continue

        t = torch.tensor(values, dtype=torch.float32)
        summary[key] = {
            "mean": float(t.mean().item()),
            "std": float(t.std(unbiased=False).item()),
            "median": float(t.median().item()),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
        }

    return summary


def save_summary_json(payload, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2)


def save_per_sample_csv(rows, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def collect_matching_samples(
    radar_root: Path,
    input_config: str,
    target_config: str,
    prefix: str,
):
    input_dir = radar_root / input_config
    target_dir = radar_root / target_config

    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input folder: {input_dir}")
    if not target_dir.exists():
        raise FileNotFoundError(f"Missing target folder: {target_dir}")

    input_samples = {p.stem for p in input_dir.glob(f"{prefix}_*.npy")}
    target_samples = {p.stem for p in target_dir.glob(f"{prefix}_*.npy")}

    common_samples = sorted(input_samples.intersection(target_samples))

    if not common_samples:
        raise RuntimeError(
            f"No common samples found for prefix={prefix} "
            f"between {input_config} and {target_config}"
        )

    return common_samples


class FilteredRadarHeatmapDataset(Dataset):
    def __init__(
        self,
        project_root: Path,
        sample_ids,
        input_config: str,
        target_config: str,
        normalize_mode: str,
        db_min: float,
        db_max: float,
    ):
        from heatmap_dataloader import normalize_heatmap

        self.project_root = Path(project_root)
        self.radar_root = self.project_root / "Processed_Dataset" / "radar_heatmaps"
        self.sample_ids = list(sample_ids)
        self.input_config = input_config
        self.target_config = target_config
        self.normalize_mode = normalize_mode
        self.db_min = db_min
        self.db_max = db_max
        self.normalize_heatmap = normalize_heatmap

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        input_path = self.radar_root / self.input_config / f"{sample_id}.npy"
        target_path = self.radar_root / self.target_config / f"{sample_id}.npy"

        x = np.load(input_path).astype(np.float32)
        y = np.load(target_path).astype(np.float32)

        x = self.normalize_heatmap(
            x,
            mode=self.normalize_mode,
            db_min=self.db_min,
            db_max=self.db_max,
        )
        y = self.normalize_heatmap(
            y,
            mode=self.normalize_mode,
            db_min=self.db_min,
            db_max=self.db_max,
        )

        return {
            "input": torch.from_numpy(x).unsqueeze(0),
            "target": torch.from_numpy(y).unsqueeze(0),
            "sample_id": sample_id,
        }


def save_comparison_figure(sample_id, inp, pred, target, save_path):
    import matplotlib.pyplot as plt

    inp = np.clip(inp, 0.0, 1.0)
    pred = np.clip(pred, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)

    err_inp = np.abs(inp - target)
    err_pred = np.abs(pred - target)
    err_vmax = max(float(err_inp.max()), float(err_pred.max()), 1e-6)

    items = [
        (inp, "Input 1x4", 0.0, 1.0),
        (pred, "Predicted high-res", 0.0, 1.0),
        (target, "Target 1x8", 0.0, 1.0),
        (err_inp, "|Input - Target|", 0.0, err_vmax),
        (err_pred, "|Pred - Target|", 0.0, err_vmax),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))

    for ax, (img, title, vmin, vmax) in zip(axes, items):
        im = ax.imshow(
            img,
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("Angle bin")
        ax.set_ylabel("Range bin")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(sample_id)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def evaluate_sequence(model, loader, device, args):
    model.eval()

    storage = {}
    storage.update(init_metric_storage("baseline"))
    storage.update(init_metric_storage("model"))

    per_sample_rows = []
    heatmap_dir = Path(args.out_dir) / "heatmaps"
    total_saved = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        print(f"Processing batch {batch_idx + 1}/{len(loader)}", flush=True)

        x = batch["input"].to(device)
        y = batch["target"].to(device)
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

        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        pred_cpu = pred.detach().cpu()

        for i, sample_id in enumerate(batch["sample_id"]):
            row = {"sample_id": str(sample_id)}

            for key, value in baseline_metrics.items():
                row[f"baseline_{key}"] = float(value[i].detach().cpu().item())

            for key, value in model_metrics.items():
                row[f"model_{key}"] = float(value[i].detach().cpu().item())

            per_sample_rows.append(row)

            save_comparison_figure(
                sample_id=str(sample_id),
                inp=x_cpu[i, 0].numpy(),
                pred=pred_cpu[i, 0].numpy(),
                target=y_cpu[i, 0].numpy(),
                save_path=heatmap_dir / f"{sample_id}.png",
            )
            total_saved += 1

    elapsed = time.time() - start_time
    print(f"Finished {len(per_sample_rows)} samples in {elapsed:.2f} seconds.", flush=True)
    print(f"Saved {total_saved} heatmap figures to {heatmap_dir}", flush=True)

    return summarize_metric_storage(storage), per_sample_rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run 1x4 -> 1x8 inference for all saved heatmaps whose sample IDs "
            "start with a given prefix and save comparison figures."
        )
    )

    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/radar_heatmap_training/checkpoints/best.pt",
    )
    parser.add_argument("--input_config", type=str, default="1x4")
    parser.add_argument("--target_config", type=str, default="1x8")
    parser.add_argument("--sequence_prefix", type=str, default="2019_04_09_cms1000")
    parser.add_argument("--batch_size", type=int, default=32)
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
    parser.add_argument("--mask_percentile", type=float, default=95.0)
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    parser.add_argument("--topk_percent", type=float, default=5.0)
    parser.add_argument("--out_dir", type=str, default="results/test_image")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional limit for debugging.",
    )

    args = parser.parse_args()

    project_root = setup_project_root(Path(args.project_root))

    from model_residual_attention import RadarHeatmapUNet

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    radar_root = project_root / "Processed_Dataset" / "radar_heatmaps"
    sample_ids = collect_matching_samples(
        radar_root=radar_root,
        input_config=args.input_config,
        target_config=args.target_config,
        prefix=args.sequence_prefix,
    )

    if args.max_samples is not None:
        sample_ids = sample_ids[:args.max_samples]

    print("\n========================================", flush=True)
    print("Radar heatmap image test", flush=True)
    print("========================================", flush=True)
    print(f"Project root    : {project_root}", flush=True)
    print(f"Checkpoint      : {checkpoint_path}", flush=True)
    print(f"Input config    : {args.input_config}", flush=True)
    print(f"Target config   : {args.target_config}", flush=True)
    print(f"Sequence prefix : {args.sequence_prefix}", flush=True)
    print(f"Matched samples : {len(sample_ids)}", flush=True)
    print(f"Device          : {device}", flush=True)

    dataset = FilteredRadarHeatmapDataset(
        project_root=project_root,
        sample_ids=sample_ids,
        input_config=args.input_config,
        target_config=args.target_config,
        normalize_mode=args.normalize_mode,
        db_min=args.db_min,
        db_max=args.db_max,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    base_channels = get_ckpt_arg(ckpt_args, "base_channels", args.base_channels)

    model = RadarHeatmapUNet(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
        use_sigmoid=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out_dir)

    summary, per_sample_rows = evaluate_sequence(
        model=model,
        loader=loader,
        device=device,
        args=args,
    )

    print_key_results(summary)

    summary_payload = {
        "sequence_prefix": args.sequence_prefix,
        "input_config": args.input_config,
        "target_config": args.target_config,
        "num_samples": len(sample_ids),
        "checkpoint": str(checkpoint_path),
        "metrics": summary,
    }

    summary_path = out_dir / f"summary_{args.sequence_prefix}_{args.input_config}_to_{args.target_config}.json"
    csv_path = out_dir / f"per_sample_{args.sequence_prefix}_{args.input_config}_to_{args.target_config}.csv"

    save_summary_json(summary_payload, summary_path)
    save_per_sample_csv(per_sample_rows, csv_path)

    print("\nSaved:", flush=True)
    print(f"  {summary_path}", flush=True)
    print(f"  {csv_path}", flush=True)
    print(f"  {out_dir / 'heatmaps'}", flush=True)


if __name__ == "__main__":
    main()
