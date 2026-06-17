# radar_processing.py
# FMCW radar processing:
# raw ADC -> range FFT -> virtual MIMO channels -> angle FFT -> radar image

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

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


def get_virtual_configuration(config_name: str) -> np.ndarray:
    """
    Get virtual-array channel indices for a given configuration.

    These configurations represent natural low-MIMO configurations.
    They do not use sparse interleaved channel selection.

    Virtual channel order:
        [TX1-RX1, TX1-RX2, TX1-RX3, TX1-RX4,
         TX2-RX1, TX2-RX2, TX2-RX3, TX2-RX4]

    Natural low-MIMO interpretation:
        1x8: full virtual MIMO reference
        1x6: first six available virtual channels
        1x4: one TX with four RX channels
        1x2: one TX with two RX channels
    """
    configurations = {
        "1x8": np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64),
        "1x6": np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
        "1x4": np.array([0, 1, 2, 3], dtype=np.int64),
        "1x2": np.array([0, 1], dtype=np.int64),
    }

    if config_name not in configurations:
        raise ValueError(
            f"Unknown configuration {config_name}. "
            f"Available configurations: {list(configurations.keys())}"
        )

    channel_indices = configurations[config_name]

    if np.any(channel_indices < 0) or np.any(
        channel_indices >= config.NUM_VIRTUAL_CHANNELS
    ):
        raise ValueError(
            f"Configuration {config_name} contains invalid virtual channel indices"
        )

    return channel_indices


def get_available_configurations() -> Tuple[str, ...]:
    """
    Return the supported virtual-array configuration names.
    """
    return ("1x8", "1x6", "1x4", "1x2")


def validate_configuration_names(config_names: Sequence[str]) -> Tuple[str, ...]:
    """
    Validate and normalize a list of configuration names.
    """
    if not config_names:
        raise ValueError("At least one configuration must be selected")

    available = set(get_available_configurations())
    normalized = []

    for config_name in config_names:
        if config_name not in available:
            raise ValueError(
                f"Unknown configuration {config_name}. "
                f"Available configurations: {list(get_available_configurations())}"
            )
        if config_name not in normalized:
            normalized.append(config_name)

    return tuple(normalized)


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


def select_virtual_configuration(
    virtual_cube: np.ndarray,
    channel_indices: np.ndarray,
) -> np.ndarray:
    """
    Select a reduced virtual-array configuration.

    Input:
        virtual_cube: [range_bins, chirps, virtual_channels]
        channel_indices: [num_selected_channels]
    """
    if virtual_cube.ndim != 3:
        raise ValueError(f"virtual_cube must be 3D, got {virtual_cube.shape}")

    channel_indices = np.asarray(channel_indices, dtype=np.int64)

    if channel_indices.ndim != 1:
        raise ValueError("channel_indices must be a 1D array")

    if np.any(channel_indices < 0) or np.any(
        channel_indices >= virtual_cube.shape[-1]
    ):
        raise ValueError(
            f"Configuration indices out of bounds for virtual cube with "
            f"{virtual_cube.shape[-1]} channels"
        )

    return virtual_cube[:, :, channel_indices]


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
    channel_indices: Optional[np.ndarray] = None,
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

    # Step 3: Select virtual-array configuration
    if channel_indices is not None:
        v_cube = select_virtual_configuration(v_cube, channel_indices)

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


def save_configuration_pair_comparison(
    left_img: np.ndarray,
    right_img: np.ndarray,
    save_path: Path,
    left_name: str,
    right_name: str,
    dynamic_range_db: float = 45.0,
) -> None:
    """
    Save a side-by-side comparison for any two configurations.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    vmin, vmax = get_common_db_scale(
        [left_img, right_img],
        dynamic_range_db=dynamic_range_db,
    )

    plt.figure(figsize=(11, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(left_img, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    plt.title(f"Configuration {left_name}")
    plt.xlabel("Angle bin")
    plt.ylabel("Range bin")
    plt.colorbar(label="dB")

    plt.subplot(1, 2, 2)
    plt.imshow(right_img, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    plt.title(f"Configuration {right_name}")
    plt.xlabel("Angle bin")
    plt.ylabel("Range bin")
    plt.colorbar(label="dB")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_configuration_comparison(
    images_by_config: Dict[str, np.ndarray],
    save_path: Path,
    dynamic_range_db: float = 45.0,
) -> None:
    """
    Save comparison across different virtual-array configurations.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    available_configs = list(images_by_config.keys())
    images = [images_by_config[name] for name in available_configs]

    vmin, vmax = get_common_db_scale(images, dynamic_range_db=dynamic_range_db)

    plt.figure(figsize=(4 * len(available_configs), 4))

    for i, config_name in enumerate(available_configs):
        plt.subplot(1, len(available_configs), i + 1)
        plt.imshow(
            images_by_config[config_name],
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )

        plt.title(config_name)
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

    preprocessing_config = validate_configuration_names(["1x8"])[0]

    # Choose which configurations to generate and compare here.
    selected_configs = validate_configuration_names(["1x8", "1x6", "1x4", "1x2"])
    comparison_pair = validate_configuration_names(["1x8", "1x4"])

    if len(comparison_pair) != 2:
        raise ValueError("comparison_pair must contain exactly two configurations")

    if not set(comparison_pair).issubset(set(selected_configs)):
        raise ValueError(
            "comparison_pair must be included in selected_configs so the images "
            "are generated before comparison"
        )

    preprocessing_channels = get_virtual_configuration(preprocessing_config)

    # ------------------------------------------------------------
    # 1. Preprocessing ablation images
    # ------------------------------------------------------------
    raw_img = make_radar_image(
        adc,
        channel_indices=preprocessing_channels,
        mode="raw",
        suppress_horizontal=False,
        use_db=True,
        normalize=False,
    )

    clutter_img = make_radar_image(
        adc,
        channel_indices=preprocessing_channels,
        mode="clutter_removed",
        suppress_horizontal=False,
        use_db=True,
        normalize=False,
    )

    final_full_img = make_radar_image(
        adc,
        channel_indices=preprocessing_channels,
        mode="clutter_removed",
        suppress_horizontal=True,
        use_db=True,
        normalize=False,
    )

    # ------------------------------------------------------------
    # 2. Full and reduced-configuration images
    # ------------------------------------------------------------
    images_by_config = {}

    for config_name in selected_configs:
        channel_indices = get_virtual_configuration(config_name)

        img = make_radar_image(
            adc,
            channel_indices=channel_indices,
            mode="clutter_removed",
            suppress_horizontal=True,
            use_db=True,
            normalize=False,
        )

        images_by_config[config_name] = img

    left_config_name, right_config_name = comparison_pair
    left_config_img = images_by_config[left_config_name]
    right_config_img = images_by_config[right_config_name]

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

    save_configuration_comparison(
        images_by_config,
        out_dir / "configuration_comparison.png",
        dynamic_range_db=45.0,
    )

    save_configuration_pair_comparison(
        left_config_img,
        right_config_img,
        out_dir / f"{left_config_name}_vs_{right_config_name}.png",
        left_name=left_config_name,
        right_name=right_config_name,
        dynamic_range_db=45.0,
    )

    print("Saved necessary visualization figures:")
    print(f"  {out_dir / 'processing_ablation.png'}")
    print(f"  {out_dir / 'configuration_comparison.png'}")
    print(f"  {out_dir / f'{left_config_name}_vs_{right_config_name}.png'}")
