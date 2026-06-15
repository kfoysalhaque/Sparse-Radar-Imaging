# radar_processing.py
# FMCW radar processing:
# raw ADC -> range FFT -> virtual MIMO channels -> angle FFT -> radar image

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import config
from radar_io import collect_all_frames, load_adc_frame


# ============================================================
# Basic utility functions
# ============================================================

def to_db(x: np.ndarray, eps: float = config.EPS) -> np.ndarray:
    """
    Convert linear power/intensity image to dB scale.
    """
    return 10.0 * np.log10(np.abs(x) + eps)


def normalize_image(img: np.ndarray, eps: float = config.EPS) -> np.ndarray:
    """
    Normalize image to [0, 1].
    """
    img = np.asarray(img)
    img_min = np.min(img)
    img_max = np.max(img)
    return (img - img_min) / (img_max - img_min + eps)


def range_wise_background_suppression(
    img: np.ndarray,
    method: str = "median",
) -> np.ndarray:
    """
    Suppress horizontal range-wise bands.

    This removes the common background level at each range bin and keeps
    angle-localized structures.
    """
    img = np.asarray(img)

    if method == "median":
        bg = np.median(img, axis=1, keepdims=True)
    elif method == "mean":
        bg = np.mean(img, axis=1, keepdims=True)
    else:
        raise ValueError("method must be 'median' or 'mean'")

    img = img - bg
    img = np.maximum(img, 0.0)
    return img


def get_common_db_scale(
    images,
    dynamic_range_db: float = 45.0,
) -> Tuple[float, float]:
    """
    Get common dB scale for fair visualization across multiple images.
    """
    vmax = max(float(np.max(img)) for img in images)
    vmin = vmax - dynamic_range_db
    return vmin, vmax


def get_fixed_mask(ratio: float) -> np.ndarray:
    """
    Get fixed sparse MIMO channel mask from config.py.
    """
    if ratio not in config.FIXED_MASKS:
        raise ValueError(
            f"Unknown sparse ratio {ratio}. "
            f"Available ratios: {list(config.FIXED_MASKS.keys())}"
        )

    mask = np.array(config.FIXED_MASKS[ratio], dtype=np.float32)

    if mask.shape[0] != config.NUM_VIRTUAL_CHANNELS:
        raise ValueError(
            f"Mask length must be {config.NUM_VIRTUAL_CHANNELS}, "
            f"but got {mask.shape[0]}"
        )

    return mask


# ============================================================
# FMCW processing functions
# ============================================================

def remove_static_clutter(adc_data: np.ndarray) -> np.ndarray:
    """
    Remove static clutter by subtracting the mean across chirps.

    Input:
        adc_data: [samples, chirps, rx, tx]

    Output:
        clutter-removed ADC data with same shape.
    """
    return adc_data - np.mean(adc_data, axis=1, keepdims=True)


def range_fft(adc_data: np.ndarray) -> np.ndarray:
    """
    Apply range FFT over ADC samples / fast-time dimension.

    Input:
        adc_data: [samples, chirps, rx, tx]

    Output:
        range_cube: [range_bins, chirps, rx, tx]
    """
    if adc_data.ndim != 4:
        raise ValueError(f"adc_data must be 4D, got shape {adc_data.shape}")

    window = np.hanning(adc_data.shape[0]).reshape(-1, 1, 1, 1)
    adc_win = adc_data * window

    range_cube = np.fft.fft(
        adc_win,
        n=config.RANGE_FFT_SIZE,
        axis=0,
    )

    return range_cube


def form_virtual_array(radar_cube: np.ndarray) -> np.ndarray:
    """
    Convert physical RX/TX dimensions into virtual MIMO channels.

    Input:
        radar_cube: [range_bins, chirps, rx, tx]

    Output:
        virtual_cube: [range_bins, chirps, virtual_channels]

    Virtual channel order:
        TX1-RX1, TX1-RX2, TX1-RX3, TX1-RX4,
        TX2-RX1, TX2-RX2, TX2-RX3, TX2-RX4
    """
    if radar_cube.ndim != 4:
        raise ValueError(f"radar_cube must be 4D, got shape {radar_cube.shape}")

    num_rx = radar_cube.shape[2]
    num_tx = radar_cube.shape[3]

    if num_rx != config.NUM_RX or num_tx != config.NUM_TX:
        raise ValueError(
            f"Expected rx={config.NUM_RX}, tx={config.NUM_TX}, "
            f"but got rx={num_rx}, tx={num_tx}"
        )

    # [range, chirp, rx, tx] -> [range, chirp, tx, rx]
    radar_cube = np.transpose(radar_cube, (0, 1, 3, 2))

    # [range, chirp, tx, rx] -> [range, chirp, tx*rx]
    virtual_cube = radar_cube.reshape(
        radar_cube.shape[0],
        radar_cube.shape[1],
        config.NUM_VIRTUAL_CHANNELS,
    )

    return virtual_cube


def apply_virtual_mask(
    virtual_cube: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Apply sparse MIMO channel mask.

    Input:
        virtual_cube: [range_bins, chirps, virtual_channels]
        mask: [virtual_channels], where 1=keep and 0=remove
    """
    if virtual_cube.ndim != 3:
        raise ValueError(f"virtual_cube must be 3D, got {virtual_cube.shape}")

    mask = np.asarray(mask, dtype=np.float32)

    if mask.shape[0] != virtual_cube.shape[-1]:
        raise ValueError(
            f"Mask length {mask.shape[0]} does not match "
            f"virtual channels {virtual_cube.shape[-1]}"
        )

    return virtual_cube * mask.reshape(1, 1, -1)


def angle_fft(
    virtual_cube: np.ndarray,
    use_window: bool = False,
) -> np.ndarray:
    """
    Apply angle FFT over virtual MIMO antenna dimension.

    Input:
        virtual_cube: [range_bins, chirps, virtual_channels]

    Output:
        angle_cube: [range_bins, chirps, angle_bins]

    Note:
        For only 8 virtual channels, antenna windowing can reduce effective
        aperture too much. Therefore, use_window=False is the default.
    """
    if virtual_cube.ndim != 3:
        raise ValueError(f"virtual_cube must be 3D, got {virtual_cube.shape}")

    if use_window:
        window = np.hanning(virtual_cube.shape[-1]).reshape(1, 1, -1)
        virtual_cube = virtual_cube * window

    angle_cube = np.fft.fft(
        virtual_cube,
        n=config.ANGLE_FFT_SIZE,
        axis=2,
    )

    angle_cube = np.fft.fftshift(angle_cube, axes=2)
    return angle_cube


def make_radar_image(
    adc_data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    mode: str = "clutter_removed",
    suppress_horizontal: bool = True,
    zero_near_range_bins: int = 2,
    use_db: bool = True,
    normalize: bool = False,
) -> np.ndarray:
    """
    Create a 2D FMCW radar image from one raw ADC frame.

    Output:
        image: [range_bins, angle_bins]

    Modes:
        raw:
            range FFT -> virtual array -> angle FFT -> average over chirps

        clutter_removed:
            subtract chirp-mean clutter -> range FFT -> virtual array
            -> angle FFT -> average over chirps
    """
    if mode not in ["raw", "clutter_removed"]:
        raise ValueError("mode must be either 'raw' or 'clutter_removed'")

    x = adc_data

    if mode == "clutter_removed":
        x = remove_static_clutter(x)

    # Step 1: Range FFT
    r_cube = range_fft(x)  # [R, C, RX, TX]

    # Step 2: Virtual MIMO array
    v_cube = form_virtual_array(r_cube)  # [R, C, V]

    # Step 3: Apply sparse MIMO mask
    if mask is not None:
        v_cube = apply_virtual_mask(v_cube, mask)

    # Step 4: Angle FFT
    a_cube = angle_fft(v_cube, use_window=False)  # [R, C, A]

    # Step 5: Non-coherent average over chirps
    power = np.abs(a_cube) ** 2
    image = np.mean(power, axis=1)  # [R, A]

    # Step 6: Remove very close-range leakage
    if zero_near_range_bins > 0:
        image[:zero_near_range_bins, :] = 0.0

    # Step 7: Suppress horizontal range-wise clutter
    if suppress_horizontal:
        image = range_wise_background_suppression(image, method="median")

    # Step 8: Convert to dB
    if use_db:
        image = to_db(image)

    # Step 9: Optional normalization
    if normalize:
        image = normalize_image(image)

    return image.astype(np.float32)


# ============================================================
# Visualization functions
# ============================================================

def save_preprocessing_ablation(
    raw_img: np.ndarray,
    clutter_img: np.ndarray,
    final_img: np.ndarray,
    save_path: Path,
    dynamic_range_db: float = 45.0,
) -> None:
    """
    Save necessary preprocessing visualization:
        raw image
        clutter-removed image
        final image after horizontal suppression
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    vmin, vmax = get_common_db_scale(
        [raw_img, clutter_img, final_img],
        dynamic_range_db=dynamic_range_db,
    )

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(raw_img, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    plt.title("Raw Chirp Average")
    plt.xlabel("Angle bin")
    plt.ylabel("Range bin")
    plt.colorbar(label="dB")

    plt.subplot(1, 3, 2)
    plt.imshow(clutter_img, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    plt.title("After Static Clutter Removal")
    plt.xlabel("Angle bin")
    plt.ylabel("Range bin")
    plt.colorbar(label="dB")

    plt.subplot(1, 3, 3)
    plt.imshow(final_img, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    plt.title("After Range-Wise Suppression")
    plt.xlabel("Angle bin")
    plt.ylabel("Range bin")
    plt.colorbar(label="dB")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_full_vs_sparse(
    full_img: np.ndarray,
    sparse_img: np.ndarray,
    save_path: Path,
    sparse_ratio: float,
    dynamic_range_db: float = 45.0,
) -> None:
    """
    Save full-channel vs one sparse-channel radar image.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    vmin, vmax = get_common_db_scale(
        [full_img, sparse_img],
        dynamic_range_db=dynamic_range_db,
    )

    plt.figure(figsize=(11, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(full_img, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    plt.title("Full 8-Channel Image")
    plt.xlabel("Angle bin")
    plt.ylabel("Range bin")
    plt.colorbar(label="dB")

    plt.subplot(1, 2, 2)
    plt.imshow(sparse_img, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    plt.title(f"Sparse Image ({int(sparse_ratio * 100)}%)")
    plt.xlabel("Angle bin")
    plt.ylabel("Range bin")
    plt.colorbar(label="dB")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_sparse_ratio_comparison(
    images_by_ratio: Dict[float, np.ndarray],
    save_path: Path,
    dynamic_range_db: float = 45.0,
) -> None:
    """
    Save comparison across different sparse MIMO channel ratios.

    Expected keys:
        1.0, 0.75, 0.5, 0.25
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ratios = sorted(images_by_ratio.keys(), reverse=True)
    images = [images_by_ratio[r] for r in ratios]

    vmin, vmax = get_common_db_scale(images, dynamic_range_db=dynamic_range_db)

    plt.figure(figsize=(4 * len(ratios), 4))

    for i, ratio in enumerate(ratios):
        plt.subplot(1, len(ratios), i + 1)
        plt.imshow(
            images_by_ratio[ratio],
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )

        if ratio == 1.0:
            title = "Full 8 Channels"
        else:
            kept = int(round(ratio * config.NUM_VIRTUAL_CHANNELS))
            title = f"{kept}/8 Channels"

        plt.title(title)
        plt.xlabel("Angle bin")
        if i == 0:
            plt.ylabel("Range bin")
        plt.colorbar(label="dB")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def select_radar_frame(
    frames: list,
    target_sequence: str,
    target_frame: str,
) -> Path:
    """
    Select one radar frame by sequence name and frame id.
    """
    for item in frames:
        if item["sequence"].name == target_sequence and item["frame_id"] == target_frame:
            return item["radar"]

    raise RuntimeError(f"Could not find {target_sequence}/{target_frame}.mat")


# ============================================================
# Self-test / visualization generation
# ============================================================

if __name__ == "__main__":
    config.create_dirs()

    frames = collect_all_frames(config.DATA_ROOT)

    # Change these two values to test another sequence/frame
    target_sequence = "2019_04_09_cms1000"
    target_frame = "000210"

    radar_path = select_radar_frame(
        frames,
        target_sequence=target_sequence,
        target_frame=target_frame,
    )

    print(f"Loading radar frame: {radar_path}")

    adc = load_adc_frame(radar_path)

    print(f"ADC shape: {adc.shape}")
    print(f"ADC dtype: {adc.dtype}")
    print(f"Complex: {np.iscomplexobj(adc)}")

    full_mask = get_fixed_mask(1.0)

    # ------------------------------------------------------------
    # 1. Preprocessing ablation images
    # ------------------------------------------------------------
    raw_img = make_radar_image(
        adc,
        mask=full_mask,
        mode="raw",
        suppress_horizontal=False,
        use_db=True,
        normalize=False,
    )

    clutter_img = make_radar_image(
        adc,
        mask=full_mask,
        mode="clutter_removed",
        suppress_horizontal=False,
        use_db=True,
        normalize=False,
    )

    final_full_img = make_radar_image(
        adc,
        mask=full_mask,
        mode="clutter_removed",
        suppress_horizontal=True,
        use_db=True,
        normalize=False,
    )

    # ------------------------------------------------------------
    # 2. Full and sparse images
    # ------------------------------------------------------------
    images_by_ratio = {}

    for ratio in [1.0, 0.75, 0.5, 0.25]:
        mask = get_fixed_mask(ratio)

        img = make_radar_image(
            adc,
            mask=mask,
            mode="clutter_removed",
            suppress_horizontal=True,
            use_db=True,
            normalize=False,
        )

        images_by_ratio[ratio] = img

    sparse_50_img = images_by_ratio[0.5]

    # ------------------------------------------------------------
    # 3. Save only necessary visualization figures
    # ------------------------------------------------------------
    out_dir = config.FIGURE_DIR / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_preprocessing_ablation(
        raw_img,
        clutter_img,
        final_full_img,
        out_dir / "processing_ablation.png",
        dynamic_range_db=45.0,
    )

    save_sparse_ratio_comparison(
        images_by_ratio,
        out_dir / "sparse_ratios_comparison.png",
        dynamic_range_db=45.0,
    )

    save_full_vs_sparse(
        final_full_img,
        sparse_50_img,
        out_dir / "full_vs_sparse_50.png",
        sparse_ratio=0.5,
        dynamic_range_db=45.0,
    )

    print("Saved necessary visualization figures:")
    print(f"  {out_dir / 'processing_ablation.png'}")
    print(f"  {out_dir / 'sparse_ratios_comparison.png'}")
    print(f"  {out_dir / 'full_vs_sparse_50.png'}")