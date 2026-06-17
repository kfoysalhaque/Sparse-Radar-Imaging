# radar_heatmap_dataset.py
# PyTorch dataset/dataloader for supervised low-MIMO radar heatmap learning.
#
# Input:
#   Processed_Dataset/radar_heatmaps/1x4/<sample_id>.npy
#
# Target:
#   Processed_Dataset/radar_heatmaps/1x8/<sample_id>.npy
#
# Example:
#   1x4 -> 1x8
#   1x2 -> 1x8
#   1x6 -> 1x8

from pathlib import Path
import argparse
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = Path("/media/foysal/Foysal-2/Github/Sparse-Radar-Imaging")


def normalize_heatmap(
    x: np.ndarray,
    mode: str = "fixed_db",
    db_min: float = 30.0,
    db_max: float = 75.0,
) -> np.ndarray:
    """
    Normalize radar heatmap for neural network training.

    mode:
        fixed_db:
            clip to [db_min, db_max] and normalize to [0, 1].
            Recommended because it matches your visualization range.

        per_sample:
            normalize each heatmap using its own min and max.

        none:
            return original dB values.
    """
    x = x.astype(np.float32)

    if mode == "fixed_db":
        x = np.clip(x, db_min, db_max)
        x = (x - db_min) / (db_max - db_min + 1e-8)

    elif mode == "per_sample":
        x_min = np.min(x)
        x_max = np.max(x)
        x = (x - x_min) / (x_max - x_min + 1e-8)

    elif mode == "none":
        pass

    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    return x.astype(np.float32)


def collect_common_samples(
    radar_root: Path,
    input_config: str,
    target_config: str,
) -> List[str]:
    """
    Find sample IDs that exist in both input_config and target_config folders.
    """
    input_dir = radar_root / input_config
    target_dir = radar_root / target_config

    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input folder: {input_dir}")

    if not target_dir.exists():
        raise FileNotFoundError(f"Missing target folder: {target_dir}")

    input_samples = set(p.stem for p in input_dir.glob("*.npy"))
    target_samples = set(p.stem for p in target_dir.glob("*.npy"))

    common_samples = sorted(input_samples.intersection(target_samples))

    if len(common_samples) == 0:
        raise RuntimeError(
            f"No common samples found between {input_config} and {target_config}"
        )

    return common_samples


def split_samples(
    samples: List[str],
    split: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> List[str]:
    """
    Deterministic random split.
    """
    if split == "all":
        return samples

    samples = list(samples)
    rng = random.Random(seed)
    rng.shuffle(samples)

    n = len(samples)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    if split == "train":
        return train_samples
    elif split == "val":
        return val_samples
    elif split == "test":
        return test_samples
    else:
        raise ValueError("split must be one of: train, val, test, all")


class RadarHeatmapPairDataset(Dataset):
    """
    Dataset for low-MIMO to full-MIMO radar heatmap learning.

    Returns:
        input  : torch.Tensor [1, 128, 64]
        target : torch.Tensor [1, 128, 64]
    """

    def __init__(
        self,
        project_root: Path = PROJECT_ROOT,
        input_config: str = "1x4",
        target_config: str = "1x8",
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        normalize_mode: str = "fixed_db",
        db_min: float = 30.0,
        db_max: float = 75.0,
    ):
        self.project_root = Path(project_root)
        self.radar_root = (
            self.project_root
            / "Processed_Dataset"
            / "radar_heatmaps"
        )

        self.input_config = input_config
        self.target_config = target_config
        self.split = split

        self.normalize_mode = normalize_mode
        self.db_min = db_min
        self.db_max = db_max

        all_samples = collect_common_samples(
            radar_root=self.radar_root,
            input_config=input_config,
            target_config=target_config,
        )

        self.samples = split_samples(
            samples=all_samples,
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split={split}")

        print(
            f"RadarHeatmapPairDataset | "
            f"{input_config} -> {target_config} | "
            f"split={split} | samples={len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]

        input_path = (
            self.radar_root
            / self.input_config
            / f"{sample_id}.npy"
        )

        target_path = (
            self.radar_root
            / self.target_config
            / f"{sample_id}.npy"
        )

        x = np.load(input_path).astype(np.float32)
        y = np.load(target_path).astype(np.float32)

        x = normalize_heatmap(
            x,
            mode=self.normalize_mode,
            db_min=self.db_min,
            db_max=self.db_max,
        )

        y = normalize_heatmap(
            y,
            mode=self.normalize_mode,
            db_min=self.db_min,
            db_max=self.db_max,
        )

        # Add channel dimension: [H, W] -> [1, H, W]
        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.from_numpy(y).unsqueeze(0)

        return {
            "input": x,
            "target": y,
            "sample_id": sample_id,
            "input_config": self.input_config,
            "target_config": self.target_config,
        }


def create_radar_heatmap_dataloaders(
    project_root: Path = PROJECT_ROOT,
    input_config: str = "1x4",
    target_config: str = "1x8",
    batch_size: int = 8,
    num_workers: int = 4,
    normalize_mode: str = "fixed_db",
    db_min: float = 30.0,
    db_max: float = 75.0,
):
    """
    Create train/val/test dataloaders.
    """
    train_set = RadarHeatmapPairDataset(
        project_root=project_root,
        input_config=input_config,
        target_config=target_config,
        split="train",
        normalize_mode=normalize_mode,
        db_min=db_min,
        db_max=db_max,
    )

    val_set = RadarHeatmapPairDataset(
        project_root=project_root,
        input_config=input_config,
        target_config=target_config,
        split="val",
        normalize_mode=normalize_mode,
        db_min=db_min,
        db_max=db_max,
    )

    test_set = RadarHeatmapPairDataset(
        project_root=project_root,
        input_config=input_config,
        target_config=target_config,
        split="test",
        normalize_mode=normalize_mode,
        db_min=db_min,
        db_max=db_max,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(
        description="Test radar heatmap dataloader."
    )

    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--input_config", type=str, default="1x4")
    parser.add_argument("--target_config", type=str, default="1x8")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--normalize_mode", type=str, default="fixed_db")

    args = parser.parse_args()

    train_loader, val_loader, test_loader = create_radar_heatmap_dataloaders(
        project_root=Path(args.project_root),
        input_config=args.input_config,
        target_config=args.target_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize_mode=args.normalize_mode,
    )

    batch = next(iter(train_loader))

    print("\nBatch check")
    print("-----------")
    print("Input shape :", batch["input"].shape)
    print("Target shape:", batch["target"].shape)
    print("Input min/max :", batch["input"].min().item(), batch["input"].max().item())
    print("Target min/max:", batch["target"].min().item(), batch["target"].max().item())
    print("Sample IDs:", batch["sample_id"][:3])


if __name__ == "__main__":
    main()