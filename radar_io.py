# radar_io.py
# Handles loading raw radar ADC .mat files and matching radar/image/label frames.

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio

import config


def find_sequence_dirs(data_root: Path) -> List[Path]:
    """
    Find all sequence folders that contain radar_raw_frame/.
    Example sequence:
        Automotive/2019_04_09_bms1000/
    """
    data_root = Path(data_root)

    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root}")

    sequence_dirs = []

    for seq_dir in sorted(data_root.iterdir()):
        if not seq_dir.is_dir():
            continue

        radar_dir = seq_dir / config.RADAR_FOLDER_NAME
        if radar_dir.exists() and radar_dir.is_dir():
            sequence_dirs.append(seq_dir)

    if len(sequence_dirs) == 0:
        raise RuntimeError(
            f"No sequence folders found under {data_root}. "
            f"Expected folders containing '{config.RADAR_FOLDER_NAME}/'."
        )

    return sequence_dirs


def find_radar_files(sequence_dir: Path) -> List[Path]:
    """
    Find all radar .mat files inside one sequence.
    """
    radar_dir = Path(sequence_dir) / config.RADAR_FOLDER_NAME

    if not radar_dir.exists():
        raise FileNotFoundError(f"Radar folder not found: {radar_dir}")

    radar_files = sorted(radar_dir.glob(f"*{config.RADAR_FILE_EXT}"))

    if len(radar_files) == 0:
        raise RuntimeError(f"No radar .mat files found in: {radar_dir}")

    return radar_files


def get_matching_paths(radar_path: Path) -> Dict[str, Optional[Path]]:
    """
    Given one radar .mat file, find matching camera image and label file.
    Handles filename padding mismatch.

    Example:
        radar_raw_frame/000003.mat
        images_0/0000000003.jpg
        text_labels/0000000003.csv
    """
    radar_path = Path(radar_path)
    seq_dir = radar_path.parent.parent
    stem = radar_path.stem

    # Convert frame id to integer, then generate possible filename styles
    frame_idx = int(stem)

    candidate_stems = [
        stem,                    # 000003
        f"{frame_idx:06d}",       # 000003
        f"{frame_idx:08d}",       # 00000003
        f"{frame_idx:09d}",       # 000000003
        f"{frame_idx:010d}",      # 0000000003
        str(frame_idx),           # 3
    ]

    image_dir = seq_dir / config.IMAGE_FOLDER_NAME
    label_dir = seq_dir / config.LABEL_FOLDER_NAME

    image_exts = [".jpg", ".jpeg", ".png"]
    label_exts = [".csv", ".txt"]

    image_path = None
    label_path = None

    # Find matching image
    if image_dir.exists():
        for cand in candidate_stems:
            for ext in image_exts:
                p = image_dir / f"{cand}{ext}"
                if p.exists():
                    image_path = p
                    break
            if image_path is not None:
                break

    # Find matching label
    if label_dir.exists():
        for cand in candidate_stems:
            for ext in label_exts:
                p = label_dir / f"{cand}{ext}"
                if p.exists():
                    label_path = p
                    break
            if label_path is not None:
                break

    return {
        "sequence": seq_dir,
        "frame_id": stem,
        "radar": radar_path,
        "image": image_path,
        "label": label_path,
    }

def collect_all_frames(data_root: Path) -> List[Dict[str, Optional[Path]]]:
    """
    Collect all radar frames across all sequence folders.

    Returns a list of dictionaries:
        {
            "sequence": Path,
            "frame_id": str,
            "radar": Path,
            "image": Path or None,
            "label": Path or None,
        }
    """
    sequence_dirs = find_sequence_dirs(data_root)

    all_frames = []

    for seq_dir in sequence_dirs:
        radar_files = find_radar_files(seq_dir)

        for radar_path in radar_files:
            all_frames.append(get_matching_paths(radar_path))

    return all_frames


def _load_mat_file(mat_path: Path) -> Dict:
    """
    Load .mat file.

    Most normal MATLAB files are handled by scipy.io.loadmat.
    If the file is MATLAB v7.3/HDF5, scipy may fail. We can add h5py support later
    if needed.
    """
    mat_path = Path(mat_path)

    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    try:
        mat = sio.loadmat(mat_path)
    except NotImplementedError as exc:
        raise NotImplementedError(
            "This .mat file seems to be MATLAB v7.3/HDF5. "
            "We need to add h5py loading support."
        ) from exc

    return mat


def inspect_mat_keys(mat_path: Path) -> None:
    """
    Print keys and array shapes inside a .mat radar file.
    Use this first to confirm the variable name and shape.
    """
    mat = _load_mat_file(mat_path)

    print(f"\nInspecting: {mat_path}")
    print("-" * 70)

    for key, value in mat.items():
        if key.startswith("__"):
            continue

        if isinstance(value, np.ndarray):
            print(
                f"Key: {key:25s} | "
                f"shape: {str(value.shape):20s} | "
                f"dtype: {value.dtype}"
            )
        else:
            print(f"Key: {key:25s} | type: {type(value)}")


def _is_possible_adc_array(arr: np.ndarray) -> bool:
    """
    Check whether an array could be the raw ADC radar data.

    Expected dataset shape:
        samples × chirps × receivers × transmitters
        128 × 255 × 4 × 2

    But we keep it flexible in case dimensions are slightly different.
    """
    if not isinstance(arr, np.ndarray):
        return False

    arr = np.squeeze(arr)

    if arr.ndim != 4:
        return False

    shape = arr.shape

    expected_dims = {
        config.NUM_ADC_SAMPLES,
        config.NUM_CHIRPS,
        config.NUM_RX,
        config.NUM_TX,
    }

    # Strong match: all expected dimensions appear somewhere.
    if all(dim in shape for dim in expected_dims):
        return True

    # Flexible match: 4D array with one small TX/RX-like dimension.
    if config.NUM_RX in shape and config.NUM_TX in shape:
        return True

    return False


def find_adc_key(mat: Dict) -> str:
    """
    Find the key corresponding to the raw ADC radar array.

    Since different .mat files may use different variable names,
    this function searches for a 4D array matching the expected radar shape.
    """
    candidate_keys = []

    for key, value in mat.items():
        if key.startswith("__"):
            continue

        if isinstance(value, np.ndarray) and _is_possible_adc_array(value):
            candidate_keys.append(key)

    if len(candidate_keys) == 0:
        available = [
            (k, v.shape, v.dtype)
            for k, v in mat.items()
            if isinstance(v, np.ndarray) and not k.startswith("__")
        ]
        raise KeyError(
            "Could not automatically find ADC radar data key.\n"
            f"Available arrays: {available}\n"
            "Run inspect_mat_keys(mat_path) and check the correct variable name."
        )

    if len(candidate_keys) > 1:
        print(f"Warning: multiple possible ADC keys found: {candidate_keys}")
        print(f"Using first candidate: {candidate_keys[0]}")

    return candidate_keys[0]


def load_adc_frame(mat_path: Path, adc_key: Optional[str] = None) -> np.ndarray:
    """
    Load one raw ADC radar frame.

    Returns:
        adc_data: np.ndarray with expected shape:
            [samples, chirps, receivers, transmitters]
            [128, 255, 4, 2]
    """
    mat = _load_mat_file(mat_path)

    if adc_key is None:
        adc_key = find_adc_key(mat)

    adc = np.array(mat[adc_key])
    adc = np.squeeze(adc)

    if adc.ndim != 4:
        raise ValueError(
            f"Loaded ADC data must be 4D, but got shape {adc.shape} "
            f"from key '{adc_key}'."
        )

    adc = ensure_adc_shape(adc)

    return adc


def ensure_adc_shape(adc: np.ndarray) -> np.ndarray:
    """
    Ensure ADC shape is:
        [samples, chirps, receivers, transmitters]

    The dataset documentation says this is the expected order:
        128 × 255 × 4 × 2

    If dimensions are permuted, this function tries to reorder them.
    """
    adc = np.asarray(adc)
    adc = np.squeeze(adc)

    target_shape = (
        config.NUM_ADC_SAMPLES,
        config.NUM_CHIRPS,
        config.NUM_RX,
        config.NUM_TX,
    )

    if adc.shape == target_shape:
        return adc

    shape = adc.shape

    if sorted(shape) != sorted(target_shape):
        raise ValueError(
            f"ADC shape {shape} does not match expected dimensions {target_shape}."
        )

    # Find axis positions for each expected dimension.
    # This assumes all target dimensions are unique: 128, 255, 4, 2.
    axis_samples = shape.index(config.NUM_ADC_SAMPLES)
    axis_chirps = shape.index(config.NUM_CHIRPS)
    axis_rx = shape.index(config.NUM_RX)
    axis_tx = shape.index(config.NUM_TX)

    adc = np.transpose(adc, (axis_samples, axis_chirps, axis_rx, axis_tx))

    if adc.shape != target_shape:
        raise RuntimeError(f"Failed to reorder ADC data. Got {adc.shape}.")

    return adc


def load_label_file(label_path: Optional[Path]) -> Optional[pd.DataFrame]:
    """
    Load one label CSV file.

    Label format:
        [uid, class, px, py, wid, len]

    Returns:
        pandas DataFrame or None if label does not exist.
    """
    if label_path is None:
        return None

    label_path = Path(label_path)

    if not label_path.exists():
        return None

    columns = ["uid", "class", "px", "py", "wid", "len"]

    try:
        df = pd.read_csv(label_path, header=None, names=columns)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=columns)

    return df


def print_dataset_summary(data_root: Path) -> None:
    """
    Print simple dataset summary.
    """
    sequence_dirs = find_sequence_dirs(data_root)
    all_frames = collect_all_frames(data_root)

    num_images = sum(1 for f in all_frames if f["image"] is not None)
    num_labels = sum(1 for f in all_frames if f["label"] is not None)

    print("\nDataset Summary")
    print("=" * 70)
    print(f"DATA_ROOT        : {data_root}")
    print(f"Sequences found  : {len(sequence_dirs)}")
    print(f"Radar frames     : {len(all_frames)}")
    print(f"Matched images   : {num_images}")
    print(f"Matched labels   : {num_labels}")
    print("=" * 70)

    print("\nFirst 5 frames:")
    for item in all_frames[:5]:
        print(
            f"seq={item['sequence'].name}, "
            f"frame={item['frame_id']}, "
            f"radar={item['radar'].name}, "
            f"image={'yes' if item['image'] else 'no'}, "
            f"label={'yes' if item['label'] else 'no'}"
        )


if __name__ == "__main__":
    # Create output directories if needed
    config.create_dirs()

    # Print dataset summary
    print_dataset_summary(config.DATA_ROOT)

    # Inspect and load the first radar frame
    frames = collect_all_frames(config.DATA_ROOT)
    first_radar_path = frames[0]["radar"]

    inspect_mat_keys(first_radar_path)

    adc = load_adc_frame(first_radar_path)

    print("\nLoaded first ADC frame")
    print("-" * 70)
    print(f"Path  : {first_radar_path}")
    print(f"Shape : {adc.shape}")
    print(f"Dtype : {adc.dtype}")
    print(f"Complex data: {np.iscomplexobj(adc)}")

    label_df = load_label_file(frames[0]["label"])
    if label_df is not None:
        print("\nFirst label file:")
        print(label_df.head())