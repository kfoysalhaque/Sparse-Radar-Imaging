# prepare_data.py
# Create training-ready full/sparse radar image pairs from raw FMCW ADC data.

from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import random

import numpy as np

import config
from radar_io import collect_all_frames, load_adc_frame
from radar_processing import make_radar_image, get_fixed_mask


# ============================================================
# Helper functions
# ============================================================

def ratio_to_tag(ratio: float) -> str:
    """
    Convert ratio to filename tag.

    Example:
        0.5  -> r050
        0.75 -> r075
        0.25 -> r025
    """
    return f"r{int(round(ratio * 100)):03d}"


def scale_pair_to_01(
    full_img: np.ndarray,
    sparse_img: np.ndarray,
    dynamic_range_db: float = 45.0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Scale full and sparse images using the same dB scale.

    This is important. We should NOT normalize full and sparse independently,
    because that hides the degradation caused by sparse MIMO channels.

    Scaling:
        vmax = max peak between full and sparse image
        vmin = vmax - dynamic_range_db
        images are clipped to [vmin, vmax] and mapped to [0, 1]
    """
    vmax = float(max(np.max(full_img), np.max(sparse_img)))
    vmin = float(vmax - dynamic_range_db)

    full_clip = np.clip(full_img, vmin, vmax)
    sparse_clip = np.clip(sparse_img, vmin, vmax)

    full_scaled = (full_clip - vmin) / (vmax - vmin + config.EPS)
    sparse_scaled = (sparse_clip - vmin) / (vmax - vmin + config.EPS)

    return (
        full_scaled.astype(np.float32),
        sparse_scaled.astype(np.float32),
        vmin,
        vmax,
    )


def split_frames(
    frames: List[Dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Dict]]:
    """
    Split radar frames into train/val/test.

    Important:
        We split by frame first, then generate multiple sparse ratios
        inside the same split. This avoids leakage where the same frame
        appears in both train and test with different sparse masks.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    frames = list(frames)
    random.Random(seed).shuffle(frames)

    n_total = len(frames)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_frames = frames[:n_train]
    val_frames = frames[n_train:n_train + n_val]
    test_frames = frames[n_train + n_val:]

    return {
        "train": train_frames,
        "val": val_frames,
        "test": test_frames,
    }


def filter_frames_by_sequence(
    frames: List[Dict],
    sequence: str = None,
) -> List[Dict]:
    """
    Optionally keep only one sequence.
    """
    if sequence is None:
        return frames

    filtered = [
        item for item in frames
        if item["sequence"].name == sequence
    ]

    if len(filtered) == 0:
        raise RuntimeError(f"No frames found for sequence: {sequence}")

    return filtered


def save_npz_pair(
    save_path: Path,
    sparse_img: np.ndarray,
    full_img: np.ndarray,
    mask: np.ndarray,
    ratio: float,
    sequence_name: str,
    frame_id: str,
    vmin: float,
    vmax: float,
) -> None:
    """
    Save one training pair.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        save_path,
        sparse=sparse_img.astype(np.float32),
        full=full_img.astype(np.float32),
        mask=mask.astype(np.float32),
        ratio=np.float32(ratio),
        sequence=sequence_name,
        frame_id=frame_id,
        vmin=np.float32(vmin),
        vmax=np.float32(vmax),
    )


def process_one_frame(
    frame_item: Dict,
    split_name: str,
    output_root: Path,
    sparse_ratios: List[float],
    dynamic_range_db: float,
    overwrite: bool,
) -> int:
    """
    Process one raw ADC frame and save full/sparse image pairs.

    Returns:
        number of saved samples.
    """
    radar_path = frame_item["radar"]
    sequence_name = frame_item["sequence"].name
    frame_id = frame_item["frame_id"]

    adc = load_adc_frame(radar_path)

    # Full 8-channel reference image.
    full_mask = get_fixed_mask(1.0)

    full_img_db = make_radar_image(
        adc,
        mask=full_mask,
        mode="clutter_removed",
        suppress_horizontal=True,
        zero_near_range_bins=2,
        use_db=True,
        normalize=False,
    )

    saved_count = 0

    for ratio in sparse_ratios:
        # We usually do not save ratio=1.0 as an input pair.
        # Full image is already the target.
        if ratio >= 1.0:
            continue

        sparse_mask = get_fixed_mask(ratio)

        sparse_img_db = make_radar_image(
            adc,
            mask=sparse_mask,
            mode="clutter_removed",
            suppress_horizontal=True,
            zero_near_range_bins=2,
            use_db=True,
            normalize=False,
        )

        full_scaled, sparse_scaled, vmin, vmax = scale_pair_to_01(
            full_img=full_img_db,
            sparse_img=sparse_img_db,
            dynamic_range_db=dynamic_range_db,
        )

        ratio_tag = ratio_to_tag(ratio)

        save_name = f"{sequence_name}_{frame_id}_{ratio_tag}.npz"
        save_path = output_root / split_name / save_name

        if save_path.exists() and not overwrite:
            continue

        save_npz_pair(
            save_path=save_path,
            sparse_img=sparse_scaled,
            full_img=full_scaled,
            mask=sparse_mask,
            ratio=ratio,
            sequence_name=sequence_name,
            frame_id=frame_id,
            vmin=vmin,
            vmax=vmax,
        )

        saved_count += 1

    return saved_count


# ============================================================
# Main preprocessing function
# ============================================================

def prepare_dataset(
    data_root: Path,
    output_root: Path,
    sparse_ratios: List[float],
    sequence: str = None,
    max_frames: int = None,
    dynamic_range_db: float = 45.0,
    overwrite: bool = False,
) -> None:
    """
    Create train/val/test full/sparse radar image pairs.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("\nCollecting radar frames...")
    frames = collect_all_frames(data_root)
    frames = filter_frames_by_sequence(frames, sequence=sequence)

    if max_frames is not None:
        frames = frames[:max_frames]

    print(f"Total frames selected: {len(frames)}")

    splits = split_frames(
        frames=frames,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        seed=config.RANDOM_SEED,
    )

    print("\nSplit summary:")
    for split_name, split_frames_list in splits.items():
        print(f"  {split_name}: {len(split_frames_list)} frames")

    total_saved = 0

    for split_name, split_frames_list in splits.items():
        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing split: {split_name}")

        for idx, frame_item in enumerate(split_frames_list):
            try:
                saved = process_one_frame(
                    frame_item=frame_item,
                    split_name=split_name,
                    output_root=output_root,
                    sparse_ratios=sparse_ratios,
                    dynamic_range_db=dynamic_range_db,
                    overwrite=overwrite,
                )
                total_saved += saved

            except Exception as exc:
                print(
                    f"[Warning] Failed frame "
                    f"{frame_item['sequence'].name}/{frame_item['frame_id']}: {exc}"
                )

            if (idx + 1) % 100 == 0:
                print(
                    f"  {split_name}: processed {idx + 1}/{len(split_frames_list)} frames"
                )

    print("\nPreprocessing complete.")
    print(f"Saved samples: {total_saved}")
    print(f"Output root: {output_root}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare full/sparse FMCW radar image pairs."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=str(config.DATA_ROOT),
        help="Path to raw Automotive dataset root.",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default=str(config.PROCESSED_ROOT),
        help="Path to save processed .npz files.",
    )

    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Optional sequence name, e.g., 2019_04_09_cms1000. "
             "If not set, all sequences are used.",
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process. "
             "Use this for debugging first.",
    )

    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.75, 0.5, 0.25],
        help="Sparse MIMO channel ratios to generate.",
    )

    parser.add_argument(
        "--dynamic_range_db",
        type=float,
        default=45.0,
        help="Dynamic range for pair-wise dB scaling.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed .npz files.",
    )

    args = parser.parse_args()

    config.create_dirs()

    prepare_dataset(
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        sparse_ratios=args.ratios,
        sequence=args.sequence,
        max_frames=args.max_frames,
        dynamic_range_db=args.dynamic_range_db,
        overwrite=args.overwrite,
    )