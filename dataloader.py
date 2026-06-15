# dataset.py
# PyTorch dataset for sparse/full FMCW radar image reconstruction.

from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config


class SparseRadarDataset(Dataset):
    """
    Dataset for loading processed sparse/full radar image pairs.

    Each .npz file should contain:
        sparse   : [range_bins, angle_bins]
        full     : [range_bins, angle_bins]
        mask     : [8]
        ratio    : sparse channel ratio
        sequence : sequence name
        frame_id : frame id
    """

    def __init__(
        self,
        root_dir: Path,
        split: str = "train",
        ratio_filter: Optional[float] = None,
    ):
        """
        Args:
            root_dir: processed dataset root, e.g., processed/
            split: train, val, or test
            ratio_filter: optional ratio to load only one sparse level.
                          Example: 0.5 loads only r050 samples.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split
        self.ratio_filter = ratio_filter

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.files = sorted(self.split_dir.glob("*.npz"))

        if ratio_filter is not None:
            ratio_tag = self._ratio_to_tag(ratio_filter)
            self.files = [f for f in self.files if ratio_tag in f.name]

        if len(self.files) == 0:
            raise RuntimeError(
                f"No .npz files found in {self.split_dir} "
                f"with ratio_filter={ratio_filter}"
            )

    @staticmethod
    def _ratio_to_tag(ratio: float) -> str:
        """
        Example:
            0.5  -> r050
            0.75 -> r075
            0.25 -> r025
        """
        return f"r{int(round(ratio * 100)):03d}"

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]

        data = np.load(path, allow_pickle=True)

        sparse = data["sparse"].astype(np.float32)  # [R, A]
        full = data["full"].astype(np.float32)      # [R, A]
        mask = data["mask"].astype(np.float32)      # [8]
        ratio = float(data["ratio"])

        sequence = str(data["sequence"])
        frame_id = str(data["frame_id"])

        # Add channel dimension for CNN:
        # [R, A] -> [1, R, A]
        sparse = torch.from_numpy(sparse).unsqueeze(0)
        full = torch.from_numpy(full).unsqueeze(0)
        mask = torch.from_numpy(mask)

        sample = {
            "sparse": sparse,
            "full": full,
            "mask": mask,
            "ratio": torch.tensor(ratio, dtype=torch.float32),
            "sequence": sequence,
            "frame_id": frame_id,
            "path": str(path),
        }

        return sample


def create_dataloader(
    split: str,
    batch_size: int,
    shuffle: bool,
    ratio_filter: Optional[float] = None,
    num_workers: int = config.NUM_WORKERS,
) -> DataLoader:
    """
    Create PyTorch dataloader.
    """
    dataset = SparseRadarDataset(
        root_dir=config.PROCESSED_ROOT,
        split=split,
        ratio_filter=ratio_filter,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return loader


def create_train_val_loaders(
    ratio_filter: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    """
    train_loader = create_dataloader(
        split="train",
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        ratio_filter=ratio_filter,
    )

    val_loader = create_dataloader(
        split="val",
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        ratio_filter=ratio_filter,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test
    train_dataset = SparseRadarDataset(
        root_dir=config.PROCESSED_ROOT,
        split="train",
        ratio_filter=None,
    )

    print(f"Number of training samples: {len(train_dataset)}")

    sample = train_dataset[0]

    print("Sample keys:", sample.keys())
    print("Sparse shape:", sample["sparse"].shape)
    print("Full shape:", sample["full"].shape)
    print("Mask shape:", sample["mask"].shape)
    print("Ratio:", sample["ratio"])
    print("Sequence:", sample["sequence"])
    print("Frame ID:", sample["frame_id"])

    train_loader = create_dataloader(
        split="train",
        batch_size=4,
        shuffle=True,
    )

    batch = next(iter(train_loader))

    print("\nBatch:")
    print("Sparse:", batch["sparse"].shape)
    print("Full:", batch["full"].shape)
    print("Mask:", batch["mask"].shape)
    print("Ratio:", batch["ratio"].shape)