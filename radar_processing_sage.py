# radar_processing_sage.py
# Low-MIMO FFT-Seeded SAGE for FMCW radar imaging.
#
# This file is self-contained and should be the ONLY file modified for SAGE.
#
# It does NOT modify:
#   config.py
#   radar_io.py
#   radar_processing.py
#   README.md
#
# Main design:
#   - 1x8 FFT reference is generated using the original radar_processing.make_radar_image()
#   - 1x4 FFT baseline is generated using natural low-end channels [0,1,2,3]
#   - 1x2 FFT baseline is generated using natural low-end channels [0,1]
#   - SAGE refines only confident peak regions and does not replace whole range rows
#
# This avoids the previous issue where 1x4 = [0,2,4,6] behaved like a sparse
# subset of a full 1x8 array and split the car-like target response.

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, List

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# User-selected project root
# ============================================================

PROJECT_ROOT = Path("/media/foysal/Foysal-2/Github/Sparse-Radar-Imaging")


# ============================================================
# Natural low-MIMO configurations
# ============================================================

def get_natural_low_mimo_config(config_name: str) -> np.ndarray:
    """
    Natural low-end MIMO configurations.

    These are NOT sparse subsets such as [0,2,4,6].
    They emulate naturally lower-MIMO radar using the first available
    RX channels from one TX group.

    Virtual order from radar_processing.py:
        [TX1-RX1, TX1-RX2, TX1-RX3, TX1-RX4,
         TX2-RX1, TX2-RX2, TX2-RX3, TX2-RX4]
    """
    configs = {
        "1x8": np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64),
        "1x4": np.array([0, 1, 2, 3], dtype=np.int64),
        "1x2": np.array([0, 1], dtype=np.int64),
    }

    if config_name not in configs:
        raise ValueError(
            f"Unknown config_name={config_name}. "
            f"Available: {list(configs.keys())}"
        )

    return configs[config_name]


def get_local_channel_positions(num_channels: int) -> np.ndarray:
    """
    Local antenna positions for natural low-MIMO.

    For natural 1x4:
        selected channels = [0,1,2,3]
        local positions   = [0,1,2,3]

    For natural 1x2:
        selected channels = [0,1]
        local positions   = [0,1]
    """
    return np.arange(num_channels, dtype=np.float32)


# ============================================================
# Path and frame utilities
# ============================================================

def setup_project_root(project_root: Path) -> Path:
    project_root = Path(project_root)

    if not project_root.exists():
        raise FileNotFoundError(f"PROJECT_ROOT does not exist: {project_root}")

    sys.path.insert(0, str(project_root))
    return project_root


def resolve_project_path(project_root: Path, maybe_relative_path: Path) -> Path:
    """
    Resolve config paths that may be relative, e.g., Path('./Dataset').
    """
    maybe_relative_path = Path(maybe_relative_path)

    if maybe_relative_path.is_absolute():
        return maybe_relative_path

    return project_root / maybe_relative_path


def find_frame(frames, sequence_name: str, frame_id: str):
    for item in frames:
        if item["sequence"].name == sequence_name and item["frame_id"] == frame_id:
            return item
    return None


# ============================================================
# Basic math utilities
# ============================================================

def to_db_local(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (10.0 * np.log10(np.abs(x) + eps)).astype(np.float32)


def make_fft_consistent_angle_grid(
    angle_fft_size: int,
    d_over_lambda: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create angle grid consistent with FFT angle bins.

    This avoids using an arbitrary linear -60 to 60 degree grid.
    """
    spatial_freq = np.fft.fftshift(np.fft.fftfreq(angle_fft_size))
    sin_theta = spatial_freq / d_over_lambda
    sin_theta = np.clip(sin_theta, -1.0, 1.0)

    angle_grid_rad = np.arcsin(sin_theta).astype(np.float32)
    angle_grid_deg = np.rad2deg(angle_grid_rad).astype(np.float32)

    return angle_grid_rad, angle_grid_deg


def steering_vector(
    theta_rad: float,
    channel_positions: np.ndarray,
    d_over_lambda: float = 0.5,
    normalize: bool = False,
) -> np.ndarray:
    """
    ULA steering vector using actual local low-MIMO channel positions.

    a(theta)[m] = exp(j * 2*pi*d_over_lambda*p_m*sin(theta))

    where p_m are channel positions, e.g.:
        1x4 -> [0,1,2,3]
        1x2 -> [0,1]
    """
    channel_positions = np.asarray(channel_positions, dtype=np.float32).reshape(-1)

    a = np.exp(
        1j
        * 2.0
        * np.pi
        * d_over_lambda
        * channel_positions
        * np.sin(theta_rad)
    )

    if normalize:
        a = a / np.sqrt(len(channel_positions))

    return a.astype(np.complex64)


def estimate_amplitude(
    a: np.ndarray,
    Y: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Estimate complex amplitude across chirps.

    Args:
        a: [M]
        Y: [M, L]

    Returns:
        alpha: [L]
    """
    denom = np.vdot(a, a).real + eps
    alpha = (a.conj().reshape(1, -1) @ Y).reshape(-1) / denom
    return alpha.astype(np.complex64)


# ============================================================
# Active range-bin and peak detection
# ============================================================

def detect_active_range_bins(
    fft_image_db: np.ndarray,
    top_percent: float = 12.0,
    row_max_percentile: float = 88.0,
    row_energy_percentile: float = 75.0,
    zero_near_range_bins: int = 2,
) -> np.ndarray:
    """
    Detect range bins likely to contain target responses.

    Uses both:
        row_max    = max over angle bins
        row_energy = mean linearized energy proxy over angle bins

    This avoids running SAGE on weak clutter rows.
    """
    if fft_image_db.ndim != 2:
        raise ValueError(f"fft_image_db must be 2D, got {fft_image_db.shape}")

    num_ranges = fft_image_db.shape[0]

    row_max = np.max(fft_image_db, axis=1)

    # Energy proxy from dB image. We only need relative ranking.
    row_energy = np.mean(10.0 ** (fft_image_db / 10.0), axis=1)

    if zero_near_range_bins > 0:
        row_max[:zero_near_range_bins] = -np.inf
        row_energy[:zero_near_range_bins] = 0.0

    valid_max = row_max[np.isfinite(row_max)]
    valid_energy = row_energy[np.isfinite(row_max)]

    if valid_max.size == 0:
        return np.array([], dtype=np.int64)

    max_thr = np.percentile(valid_max, row_max_percentile)
    energy_thr = np.percentile(valid_energy, row_energy_percentile)

    candidates = np.where((row_max >= max_thr) & (row_energy >= energy_thr))[0]

    max_bins = max(1, int(round(num_ranges * top_percent / 100.0)))

    if len(candidates) > max_bins:
        score = row_max[candidates]
        order = np.argsort(score)[::-1]
        candidates = candidates[order[:max_bins]]

    return np.sort(candidates).astype(np.int64)


def select_peak_indices(
    row_db: np.ndarray,
    max_peaks: int,
    min_separation_bins: int = 3,
) -> List[int]:
    """
    Select strongest angular peak bins with simple non-maximum suppression.
    """
    row_db = np.asarray(row_db).reshape(-1)

    sorted_idx = np.argsort(row_db)[::-1]
    selected = []

    for idx in sorted_idx:
        idx = int(idx)

        if any(abs(idx - s) < min_separation_bins for s in selected):
            continue

        selected.append(idx)

        if len(selected) >= max_peaks:
            break

    return selected


def initial_angles_from_fft_row(
    fft_row_db: np.ndarray,
    angle_grid_rad: np.ndarray,
    max_targets: int,
    min_separation_bins: int = 3,
) -> Tuple[np.ndarray, List[int]]:
    peak_bins = select_peak_indices(
        row_db=fft_row_db,
        max_peaks=max_targets,
        min_separation_bins=min_separation_bins,
    )

    init_angles = angle_grid_rad[peak_bins].astype(np.float32)

    return init_angles, peak_bins


# ============================================================
# SAGE core
# ============================================================

def local_angle_candidates_from_grid(
    center_angle_rad: float,
    angle_grid_rad: np.ndarray,
    local_search_bins: int = 4,
) -> np.ndarray:
    """
    Use local candidates from the FFT-consistent angle grid.
    """
    nearest_idx = int(np.argmin(np.abs(angle_grid_rad - center_angle_rad)))

    lo = max(0, nearest_idx - local_search_bins)
    hi = min(len(angle_grid_rad), nearest_idx + local_search_bins + 1)

    return angle_grid_rad[lo:hi]


def sage_refine_one_range_bin(
    Y: np.ndarray,
    angle_grid_rad: np.ndarray,
    init_angles_rad: np.ndarray,
    channel_positions: np.ndarray,
    num_iterations: int = 4,
    local_search_bins: int = 4,
    d_over_lambda: float = 0.5,
    max_targets: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    SAGE-like angular refinement for one range bin.

    Args:
        Y:
            [num_channels, num_chirps]

    Returns:
        spectrum_power:
            sparse angular spectrum over angle_grid_rad

        info:
            refined target info
    """
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D [channels, chirps], got {Y.shape}")

    num_channels, num_chirps = Y.shape

    channel_positions = np.asarray(channel_positions, dtype=np.float32).reshape(-1)

    if len(channel_positions) != num_channels:
        raise ValueError(
            f"channel_positions length {len(channel_positions)} does not match "
            f"num_channels {num_channels}"
        )

    init_angles_rad = np.asarray(init_angles_rad, dtype=np.float32).reshape(-1)

    if max_targets is not None:
        init_angles_rad = init_angles_rad[:max_targets]

    # Conservative target count for low-MIMO.
    # For 1x2, do not try to fit too many scatterers.
    K = min(len(init_angles_rad), num_channels)

    if K == 0:
        return np.zeros(len(angle_grid_rad), dtype=np.float32), {
            "angles_rad": np.array([], dtype=np.float32),
            "angles_deg": np.array([], dtype=np.float32),
            "powers": np.array([], dtype=np.float32),
            "num_targets": 0,
            "peak_bins": [],
        }

    angles = init_angles_rad[:K].copy()
    amplitudes = np.zeros((K, num_chirps), dtype=np.complex64)

    # Initial amplitude estimates
    for k in range(K):
        a = steering_vector(
            theta_rad=float(angles[k]),
            channel_positions=channel_positions,
            d_over_lambda=d_over_lambda,
        )
        amplitudes[k, :] = estimate_amplitude(a, Y)

    # Iterative refinement
    for _ in range(num_iterations):
        for k in range(K):
            Y_others = np.zeros_like(Y, dtype=np.complex64)

            for j in range(K):
                if j == k:
                    continue

                a_j = steering_vector(
                    theta_rad=float(angles[j]),
                    channel_positions=channel_positions,
                    d_over_lambda=d_over_lambda,
                )

                Y_others += a_j.reshape(-1, 1) * amplitudes[j].reshape(1, -1)

            residual = Y - Y_others

            candidates = local_angle_candidates_from_grid(
                center_angle_rad=float(angles[k]),
                angle_grid_rad=angle_grid_rad,
                local_search_bins=local_search_bins,
            )

            best_score = -np.inf
            best_angle = float(angles[k])
            best_alpha = amplitudes[k, :]

            for theta in candidates:
                a_theta = steering_vector(
                    theta_rad=float(theta),
                    channel_positions=channel_positions,
                    d_over_lambda=d_over_lambda,
                )

                alpha_theta = estimate_amplitude(a_theta, residual)

                matched = a_theta.conj().reshape(1, -1) @ residual
                score = float(np.mean(np.abs(matched) ** 2))

                if score > best_score:
                    best_score = score
                    best_angle = float(theta)
                    best_alpha = alpha_theta

            angles[k] = best_angle
            amplitudes[k, :] = best_alpha

    powers = np.mean(np.abs(amplitudes) ** 2, axis=1).astype(np.float32)

    spectrum_power = np.zeros(len(angle_grid_rad), dtype=np.float32)
    peak_bins = []

    for k in range(K):
        peak_idx = int(np.argmin(np.abs(angle_grid_rad - angles[k])))
        spectrum_power[peak_idx] += powers[k]
        peak_bins.append(peak_idx)

    info = {
        "angles_rad": angles.astype(np.float32),
        "angles_deg": np.rad2deg(angles).astype(np.float32),
        "powers": powers.astype(np.float32),
        "num_targets": int(K),
        "peak_bins": peak_bins,
    }

    return spectrum_power, info


# ============================================================
# Safe SAGE overlay
# ============================================================

def apply_safe_sage_overlay(
    fft_row_db: np.ndarray,
    sage_spectrum_power: np.ndarray,
    init_peak_bins: Sequence[int],
    refined_peak_bins: Sequence[int],
    max_peak_shift_bins: int = 5,
    overlay_half_width_bins: int = 2,
    min_peak_gain_db: float = -6.0,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, bool, str]:
    """
    Safely overlay SAGE peaks onto the original FFT row.

    It does NOT replace the whole row.

    Reject SAGE if:
        - refined peak moves too far from FFT peak
        - SAGE peak is too weak
        - no valid peak is found
    """
    fft_row_db = fft_row_db.astype(np.float32).copy()
    sage_spectrum_power = sage_spectrum_power.astype(np.float32)

    if len(refined_peak_bins) == 0 or len(init_peak_bins) == 0:
        return fft_row_db, False, "no_peak"

    fft_main_peak = int(init_peak_bins[0])
    sage_main_peak = int(refined_peak_bins[0])

    if abs(sage_main_peak - fft_main_peak) > max_peak_shift_bins:
        return fft_row_db, False, "peak_shift_too_large"

    sage_row_db = to_db_local(sage_spectrum_power, eps=eps)

    fft_peak_db = float(np.max(fft_row_db))
    sage_peak_db = float(np.max(sage_row_db))

    if not np.isfinite(sage_peak_db):
        return fft_row_db, False, "invalid_sage_peak"

    # Align SAGE peak level to FFT peak level.
    sage_row_db = sage_row_db + (fft_peak_db - sage_peak_db)

    # If still too weak after alignment logic, reject.
    if np.max(sage_row_db) < fft_peak_db + min_peak_gain_db:
        return fft_row_db, False, "sage_too_weak"

    out = fft_row_db.copy()

    for p in refined_peak_bins:
        p = int(p)

        lo = max(0, p - overlay_half_width_bins)
        hi = min(len(out), p + overlay_half_width_bins + 1)

        # Keep FFT background but reinforce/refine local SAGE peak.
        out[lo:hi] = np.maximum(out[lo:hi], sage_row_db[lo:hi])

    return out.astype(np.float32), True, "accepted"


def make_sage_image_from_cube(
    V_low: np.ndarray,
    coarse_fft_image: np.ndarray,
    channel_positions: np.ndarray,
    angle_fft_size: int,
    max_targets_per_range: int = 2,
    num_iterations: int = 6,
    local_search_bins: int = 6,
    top_range_percent: float = 20.0,
    row_max_percentile: float = 82.0,
    row_energy_percentile: float = 65.0,
    d_over_lambda: float = 0.5,
    max_peak_shift_bins: int = 8,
    overlay_half_width_bins: int = 4,
) -> Tuple[np.ndarray, Dict]:
    """
    Generate SAGE-refined image from low-MIMO cube.

    This starts from the FFT image and only overlays confident local SAGE peaks.
    """
    if V_low.ndim != 3:
        raise ValueError(f"V_low must be [range, chirp, channel], got {V_low.shape}")

    if coarse_fft_image.ndim != 2:
        raise ValueError(f"coarse_fft_image must be 2D, got {coarse_fft_image.shape}")

    angle_grid_rad, angle_grid_deg = make_fft_consistent_angle_grid(
        angle_fft_size=angle_fft_size,
        d_over_lambda=d_over_lambda,
    )

    if coarse_fft_image.shape[1] != len(angle_grid_rad):
        raise ValueError(
            f"coarse_fft_image width {coarse_fft_image.shape[1]} does not match "
            f"angle grid size {len(angle_grid_rad)}"
        )

    active_bins = detect_active_range_bins(
        fft_image_db=coarse_fft_image,
        top_percent=top_range_percent,
        row_max_percentile=row_max_percentile,
        row_energy_percentile=row_energy_percentile,
        zero_near_range_bins=2,
    )

    sage_image = coarse_fft_image.copy().astype(np.float32)

    accepted_rows = []
    rejected_rows = {}

    range_infos = {}

    for r in active_bins:
        r = int(r)

        fft_row = coarse_fft_image[r, :]

        init_angles, init_peak_bins = initial_angles_from_fft_row(
            fft_row_db=fft_row,
            angle_grid_rad=angle_grid_rad,
            max_targets=max_targets_per_range,
            min_separation_bins=3,
        )

        Y = V_low[r, :, :].T
        # [channels, chirps]

        sage_power, r_info = sage_refine_one_range_bin(
            Y=Y,
            angle_grid_rad=angle_grid_rad,
            init_angles_rad=init_angles,
            channel_positions=channel_positions,
            num_iterations=num_iterations,
            local_search_bins=local_search_bins,
            d_over_lambda=d_over_lambda,
            max_targets=max_targets_per_range,
        )

        updated_row, accepted, reason = apply_safe_sage_overlay(
            fft_row_db=fft_row,
            sage_spectrum_power=sage_power,
            init_peak_bins=init_peak_bins,
            refined_peak_bins=r_info["peak_bins"],
            max_peak_shift_bins=max_peak_shift_bins,
            overlay_half_width_bins=overlay_half_width_bins,
        )

        if accepted:
            sage_image[r, :] = updated_row
            accepted_rows.append(r)
        else:
            rejected_rows[r] = reason

        r_info["init_peak_bins"] = init_peak_bins
        r_info["accepted"] = accepted
        r_info["reject_reason"] = reason
        range_infos[r] = r_info

    info = {
        "active_range_bins": active_bins,
        "accepted_rows": np.array(accepted_rows, dtype=np.int64),
        "rejected_rows": rejected_rows,
        "num_active_range_bins": int(len(active_bins)),
        "num_accepted_rows": int(len(accepted_rows)),
        "angle_grid_deg": angle_grid_deg,
        "angle_grid_rad": angle_grid_rad,
        "range_infos": range_infos,
        "params": {
            "angle_fft_size": int(angle_fft_size),
            "max_targets_per_range": int(max_targets_per_range),
            "num_iterations": int(num_iterations),
            "local_search_bins": int(local_search_bins),
            "top_range_percent": float(top_range_percent),
            "row_max_percentile": float(row_max_percentile),
            "row_energy_percentile": float(row_energy_percentile),
            "d_over_lambda": float(d_over_lambda),
            "max_peak_shift_bins": int(max_peak_shift_bins),
            "overlay_half_width_bins": int(overlay_half_width_bins),
        },
    }

    return sage_image.astype(np.float32), info


# ============================================================
# Metrics and plotting
# ============================================================

def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff ** 2))


def get_common_display_limits(images: Dict[str, np.ndarray], dynamic_range_db: float = 45.0):
    vmax = max(float(np.max(img)) for img in images.values())
    vmin = vmax - dynamic_range_db
    return vmin, vmax


def save_comparison_figure(
    images: Dict[str, np.ndarray],
    save_path: Path,
    title: str,
    dynamic_range_db: float = 45.0,
    cmap: str = "viridis",
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    vmin, vmax = get_common_display_limits(images, dynamic_range_db=dynamic_range_db)

    names = list(images.keys())

    plt.figure(figsize=(4.3 * len(names), 4.2))

    for i, name in enumerate(names):
        plt.subplot(1, len(names), i + 1)
        plt.imshow(
            images[name],
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        plt.title(name)
        plt.xlabel("Angle bin")
        if i == 0:
            plt.ylabel("Range bin")
        plt.colorbar(label="dB")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_difference_figure(
    images: Dict[str, np.ndarray],
    reference_key: str,
    save_path: Path,
    title: str = "Difference from 1x8 FFT",
    cmap: str = "viridis",
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ref = images[reference_key]

    diff_images = {}

    for key, img in images.items():
        if key == reference_key:
            continue
        diff_images[f"{reference_key} - {key}"] = np.abs(ref - img).astype(np.float32)

    all_vals = np.concatenate([x.reshape(-1) for x in diff_images.values()])
    vmax = float(np.percentile(all_vals, 99.0))
    vmax = max(vmax, 1e-6)

    names = list(diff_images.keys())

    plt.figure(figsize=(4.3 * len(names), 4.2))

    for i, name in enumerate(names):
        plt.subplot(1, len(names), i + 1)
        plt.imshow(
            diff_images[name],
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=vmax,
            cmap=cmap,
        )
        plt.title(name)
        plt.xlabel("Angle bin")
        if i == 0:
            plt.ylabel("Range bin")
        plt.colorbar(label="Abs. diff. dB")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_metrics(
    metrics: Dict[str, float],
    sage_info_1x4: Dict,
    sage_info_1x2: Dict,
    save_path: Path,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        f.write("Low-MIMO FFT-Seeded SAGE Metrics\n")
        f.write("================================\n\n")

        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")

        f.write("\nImprovement summary\n")
        f.write("-------------------\n")
        f.write(
            f"1x4 SAGE improved MAE: "
            f"{metrics['MAE_1x8_vs_1x4_SAGE'] < metrics['MAE_1x8_vs_1x4_FFT']}\n"
        )
        f.write(
            f"metrics['MAE_1x8_vs_11x2 SAGE improved MAE: "
            f"{metrics['MAE_1x8_vs_1x2_SAGE'] < metrics['MAE_1x8_vs_1x2_FFT']}\n"
        )

        f.write("\nSAGE 1x4 info\n")
        f.write("-------------\n")
        f.write(f"Active rows: {sage_info_1x4['num_active_range_bins']}\n")
        f.write(f"Accepted rows: {sage_info_1x4['num_accepted_rows']}\n")
        f.write(f"Accepted row indices: {sage_info_1x4['accepted_rows'].tolist()}\n")

        f.write("\nSAGE 1x2 info\n")
        f.write("-------------\n")
        f.write(f"Active rows: {sage_info_1x2['num_active_range_bins']}\n")
        f.write(f"Accepted rows: {sage_info_1x2['num_accepted_rows']}\n")
        f.write(f"Accepted row indices: {sage_info_1x2['accepted_rows'].tolist()}\n")


def print_sage_info(name: str, info: Dict) -> None:
    print(f"\n{name} SAGE info")
    print("-" * (len(name) + 10))
    print(f"Active rows  : {info['num_active_range_bins']}")
    print(f"Accepted rows: {info['num_accepted_rows']}")
    print(f"Accepted row indices: {info['accepted_rows'][:20].tolist()}")

    if len(info["rejected_rows"]) > 0:
        preview = list(info["rejected_rows"].items())[:10]
        print(f"Rejected preview: {preview}")


# ============================================================
# Main runner
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Low-MIMO FFT-Seeded SAGE on one real radar frame."
    )

    parser.add_argument(
        "--project_root",
        type=str,
        default=str(PROJECT_ROOT),
        help="Path to Sparse-Radar-Imaging folder.",
    )

    parser.add_argument(
        "--sequence",
        type=str,
        default="2019_04_09_cms1000",
        help="Radar sequence name.",
    )

    parser.add_argument(
        "--frame",
        type=str,
        default="000210",
        help="Radar frame id.",
    )

    parser.add_argument(
        "--num_iterations",
        type=int,
        default=4,
        help="Number of SAGE iterations.",
    )

    parser.add_argument(
        "--top_range_percent",
        type=float,
        default=12.0,
        help="Percentage of range bins considered for SAGE.",
    )

    args = parser.parse_args()

    project_root = setup_project_root(Path(args.project_root))

    import config
    import radar_processing
    from radar_io import collect_all_frames, load_adc_frame

    config.create_dirs()

    data_root = resolve_project_path(project_root, config.DATA_ROOT)

    print("\n========================================")
    print("Low-MIMO FFT-Seeded SAGE")
    print("========================================")
    print(f"Project root : {project_root}")
    print(f"Dataset root : {data_root}")
    print(f"Sequence     : {args.sequence}")
    print(f"Frame        : {args.frame}")
    print(f"Angle FFT    : {config.ANGLE_FFT_SIZE}")

    # ------------------------------------------------------------
    # 1. Load real radar frame
    # ------------------------------------------------------------
    frames = collect_all_frames(data_root)

    frame_item = find_frame(
        frames=frames,
        sequence_name=args.sequence,
        frame_id=args.frame,
    )

    if frame_item is None:
        raise RuntimeError(
            f"Could not find frame {args.frame} in sequence {args.sequence}"
        )

    radar_path = frame_item["radar"]

    print("\nSelected frame")
    print("--------------")
    print(f"Radar path: {radar_path}")
    print(f"Image path: {frame_item.get('image')}")
    print(f"Label path: {frame_item.get('label')}")

    adc = load_adc_frame(radar_path)

    print("\nADC")
    print("---")
    print(f"Shape   : {adc.shape}")
    print(f"Dtype   : {adc.dtype}")
    print(f"Complex : {np.iscomplexobj(adc)}")

    # ------------------------------------------------------------
    # 2. Natural low-MIMO channel configs
    # ------------------------------------------------------------
    ch_1x8 = get_natural_low_mimo_config("1x8")
    ch_1x4 = get_natural_low_mimo_config("1x4")
    ch_1x2 = get_natural_low_mimo_config("1x2")

    print("\nNatural low-MIMO configs")
    print("------------------------")
    print(f"1x8: {ch_1x8.tolist()}")
    print(f"1x4: {ch_1x4.tolist()}")
    print(f"1x2: {ch_1x2.tolist()}")

    # ------------------------------------------------------------
    # 3. Correct FFT images using original radar_processing.py
    # ------------------------------------------------------------
    img_1x8_fft = radar_processing.make_radar_image(
        adc,
        channel_indices=ch_1x8,
        mode="clutter_removed",
        suppress_horizontal=True,
        zero_near_range_bins=2,
        use_db=True,
        normalize=False,
    )

    img_1x4_fft = radar_processing.make_radar_image(
        adc,
        channel_indices=ch_1x4,
        mode="clutter_removed",
        suppress_horizontal=True,
        zero_near_range_bins=2,
        use_db=True,
        normalize=False,
    )

    img_1x2_fft = radar_processing.make_radar_image(
        adc,
        channel_indices=ch_1x2,
        mode="clutter_removed",
        suppress_horizontal=True,
        zero_near_range_bins=2,
        use_db=True,
        normalize=False,
    )

    print("\nFFT image shapes")
    print("----------------")
    print(f"1x8 FFT: {img_1x8_fft.shape}")
    print(f"1x4 FFT: {img_1x4_fft.shape}")
    print(f"1x2 FFT: {img_1x2_fft.shape}")

    # ------------------------------------------------------------
    # 4. Build virtual cube for SAGE input
    # ------------------------------------------------------------
    adc_proc = radar_processing.remove_static_clutter(adc)
    r_cube = radar_processing.range_fft(adc_proc)
    V_full = radar_processing.form_virtual_array(r_cube).astype(np.complex64)

    V_1x4 = radar_processing.select_virtual_configuration(V_full, ch_1x4)
    V_1x2 = radar_processing.select_virtual_configuration(V_full, ch_1x2)

    pos_1x4 = get_local_channel_positions(len(ch_1x4))
    pos_1x2 = get_local_channel_positions(len(ch_1x2))

    print("\nVirtual cubes")
    print("-------------")
    print(f"V_full: {V_full.shape}")
    print(f"V_1x4: {V_1x4.shape}, positions={pos_1x4.tolist()}")
    print(f"V_1x2: {V_1x2.shape}, positions={pos_1x2.tolist()}")

    # ------------------------------------------------------------
    # 5. SAGE refinement
    # ------------------------------------------------------------
    img_1x4_sage, info_1x4 = make_sage_image_from_cube(
        V_low=V_1x4,
        coarse_fft_image=img_1x4_fft,
        channel_positions=pos_1x4,
        angle_fft_size=config.ANGLE_FFT_SIZE,
        max_targets_per_range=2,
        num_iterations=args.num_iterations,
        local_search_bins=4,
        top_range_percent=args.top_range_percent,
        row_max_percentile=88.0,
        row_energy_percentile=75.0,
        d_over_lambda=0.5,
        max_peak_shift_bins=5,
        overlay_half_width_bins=2,
    )

    img_1x2_sage, info_1x2 = make_sage_image_from_cube(
        V_low=V_1x2,
        coarse_fft_image=img_1x2_fft,
        channel_positions=pos_1x2,
        angle_fft_size=config.ANGLE_FFT_SIZE,
        max_targets_per_range=1,
        num_iterations=args.num_iterations,
        local_search_bins=4,
        top_range_percent=args.top_range_percent,
        row_max_percentile=88.0,
        row_energy_percentile=75.0,
        d_over_lambda=0.5,
        max_peak_shift_bins=5,
        overlay_half_width_bins=2,
    )

    print_sage_info("1x4", info_1x4)
    print_sage_info("1x2", info_1x2)

    # ------------------------------------------------------------
    # 6. Metrics
    # ------------------------------------------------------------
    metrics = {
        "MAE_1x8_vs_1x4_FFT": mae(img_1x8_fft, img_1x4_fft),
        "MAE_1x8_vs_1x4_SAGE": mae(img_1x8_fft, img_1x4_sage),
        "MAE_1x8_vs_1x2_FFT": mae(img_1x8_fft, img_1x2_fft),
        "MAE_1x8_vs_1x2_SAGE": mae(img_1x8_fft, img_1x2_sage),
        "MSE_1x8_vs_1x4_FFT": mse(img_1x8_fft, img_1x4_fft),
        "MSE_1x8_vs_1x4_SAGE": mse(img_1x8_fft, img_1x4_sage),
        "MSE_1x8_vs_1x2_FFT": mse(img_1x8_fft, img_1x2_fft),
        "MSE_1x8_vs_1x2_SAGE": mse(img_1x8_fft, img_1x2_sage),
    }

    print("\nMetrics")
    print("-------")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    print(
        f"\n1x4 SAGE improved MAE: "
        f"{metrics['MAE_1x8_vs_1x4_SAGE'] < metrics['MAE_1x8_vs_1x4_FFT']}"
    )
    print(
        f"1x2 SAGE improved MAE: "
        f"{metrics['MAE_1x8_vs_1x2_SAGE'] < metrics['MAE_1x8_vs_1x2_FFT']}"
    )

    # ------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------
    out_dir = project_root / "results" / "sage_debug" / args.sequence
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = out_dir / f"{args.frame}_sage_comparison.png"
    difference_path = out_dir / f"{args.frame}_sage_difference_from_1x8.png"
    metrics_path = out_dir / f"{args.frame}_sage_metrics.txt"

    images = {
        "1x8 FFT": img_1x8_fft,
        "1x4 FFT": img_1x4_fft,
        "1x4 FFT + SAGE": img_1x4_sage,
        "1x2 FFT": img_1x2_fft,
        "1x2 FFT + SAGE": img_1x2_sage,
    }

    save_comparison_figure(
        images=images,
        save_path=comparison_path,
        title="Natural Low-MIMO FFT-Seeded SAGE",
        dynamic_range_db=45.0,
        cmap="viridis",
    )

    save_difference_figure(
        images=images,
        reference_key="1x8 FFT",
        save_path=difference_path,
        title="Difference from 1x8 FFT Reference",
        cmap="viridis",
    )

    save_metrics(
        metrics=metrics,
        sage_info_1x4=info_1x4,
        sage_info_1x2=info_1x2,
        save_path=metrics_path,
    )

    print("\nSaved")
    print("-----")
    print(f"Comparison: {comparison_path}")
    print(f"Difference: {difference_path}")
    print(f"Metrics   : {metrics_path}")


if __name__ == "__main__":
    main()