# radar_processing_omp.py
# Low-MIMO FFT-Seeded ROI-OMP for FMCW radar imaging.
#
# This file does NOT modify the existing codebase.
# It does NOT reconstruct missing virtual antenna channels.
# It uses natural low-MIMO configurations:
#   1x8 = [0,1,2,3,4,5,6,7]
#   1x4 = [0,1,2,3]
#   1x2 = [0,1]
#
# Main idea:
#   1. Generate correct FFT images using radar_processing.make_radar_image().
#   2. Detect active range bins from the low-MIMO FFT image.
#   3. Run multi-snapshot OMP only on those range bins.
#   4. Blend/overlay OMP-refined angular components back into the FFT image.

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

    Virtual channel order:
        [TX1-RX1, TX1-RX2, TX1-RX3, TX1-RX4,
         TX2-RX1, TX2-RX2, TX2-RX3, TX2-RX4]

    Natural low-MIMO:
        1x4 = TX1-RX1 to TX1-RX4
        1x2 = TX1-RX1 to TX1-RX2
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

    1x4 -> [0,1,2,3]
    1x2 -> [0,1]
    """
    return np.arange(num_channels, dtype=np.float32)


# ============================================================
# Path / frame utilities
# ============================================================

def setup_project_root(project_root: Path) -> Path:
    project_root = Path(project_root)

    if not project_root.exists():
        raise FileNotFoundError(f"PROJECT_ROOT does not exist: {project_root}")

    sys.path.insert(0, str(project_root))
    return project_root


def resolve_project_path(project_root: Path, maybe_relative_path: Path) -> Path:
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


def db_to_linear(x_db: np.ndarray) -> np.ndarray:
    return (10.0 ** (x_db / 10.0)).astype(np.float32)


def linear_to_db(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (10.0 * np.log10(np.maximum(x, 0.0) + eps)).astype(np.float32)


def make_fft_consistent_angle_grid(
    angle_fft_size: int,
    d_over_lambda: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FFT-consistent angle grid.

    The angle bins correspond to fftshift(fftfreq(N)).
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
    normalize: bool = True,
) -> np.ndarray:
    """
    ULA steering vector using local natural low-MIMO channel positions.

    a(theta)[m] = exp(j * 2*pi*d_over_lambda*p_m*sin(theta))
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


def steering_dictionary(
    angle_grid_rad: np.ndarray,
    channel_positions: np.ndarray,
    d_over_lambda: float = 0.5,
    normalize: bool = True,
) -> np.ndarray:
    """
    Build angular dictionary.

    Output:
        A: [num_channels, num_angles]
    """
    A = np.stack(
        [
            steering_vector(
                theta_rad=float(theta),
                channel_positions=channel_positions,
                d_over_lambda=d_over_lambda,
                normalize=normalize,
            )
            for theta in angle_grid_rad
        ],
        axis=1,
    )

    return A.astype(np.complex64)


# ============================================================
# Active range-bin detection
# ============================================================

def detect_active_range_bins(
    fft_image_db: np.ndarray,
    top_percent: float = 18.0,
    row_max_percentile: float = 82.0,
    row_energy_percentile: float = 65.0,
    zero_near_range_bins: int = 2,
) -> np.ndarray:
    """
    Detect target-bearing range bins from a low-MIMO FFT image.
    """
    if fft_image_db.ndim != 2:
        raise ValueError(f"fft_image_db must be 2D, got {fft_image_db.shape}")

    num_ranges = fft_image_db.shape[0]

    row_max = np.max(fft_image_db, axis=1)
    row_energy = np.mean(db_to_linear(fft_image_db), axis=1)

    if zero_near_range_bins > 0:
        row_max[:zero_near_range_bins] = -np.inf
        row_energy[:zero_near_range_bins] = 0.0

    valid = np.isfinite(row_max)

    if not np.any(valid):
        return np.array([], dtype=np.int64)

    max_thr = np.percentile(row_max[valid], row_max_percentile)
    energy_thr = np.percentile(row_energy[valid], row_energy_percentile)

    candidates = np.where((row_max >= max_thr) & (row_energy >= energy_thr))[0]

    max_bins = max(1, int(round(num_ranges * top_percent / 100.0)))

    if len(candidates) > max_bins:
        scores = row_max[candidates]
        order = np.argsort(scores)[::-1]
        candidates = candidates[order[:max_bins]]

    return np.sort(candidates).astype(np.int64)


# ============================================================
# Multi-snapshot OMP
# ============================================================

def suppress_nearby_scores(
    scores: np.ndarray,
    selected: Sequence[int],
    min_separation_bins: int,
) -> np.ndarray:
    """
    Prevent selecting atoms too close to already selected atoms.
    """
    scores = scores.copy()

    for s in selected:
        lo = max(0, s - min_separation_bins)
        hi = min(len(scores), s + min_separation_bins + 1)
        scores[lo:hi] = -np.inf

    return scores


def multi_snapshot_omp(
    Y: np.ndarray,
    A: np.ndarray,
    max_atoms: int = 3,
    residual_tol: float = 0.08,
    min_separation_bins: int = 3,
) -> Tuple[List[int], np.ndarray, Dict]:
    """
    Multi-snapshot OMP / SOMP for one range bin.

    Args:
        Y:
            Low-MIMO measurements, shape [num_channels, num_chirps]

        A:
            Steering dictionary, shape [num_channels, num_angles]

    Returns:
        support:
            Selected angle-grid indices.

        X:
            Sparse coefficients for selected atoms, shape [num_selected, num_chirps].

        info:
            Debug information.
    """
    if Y.ndim != 2:
        raise ValueError(f"Y must be [channels, chirps], got {Y.shape}")

    if A.ndim != 2:
        raise ValueError(f"A must be [channels, angles], got {A.shape}")

    if A.shape[0] != Y.shape[0]:
        raise ValueError(
            f"A and Y channel dimensions mismatch: A={A.shape}, Y={Y.shape}"
        )

    residual = Y.copy().astype(np.complex64)
    support: List[int] = []

    initial_energy = float(np.linalg.norm(Y, "fro") ** 2) + 1e-8
    residual_history = []

    X = np.zeros((0, Y.shape[1]), dtype=np.complex64)

    for _ in range(max_atoms):
        corr = A.conj().T @ residual
        scores = np.sum(np.abs(corr) ** 2, axis=1)

        scores = suppress_nearby_scores(
            scores=scores,
            selected=support,
            min_separation_bins=min_separation_bins,
        )

        best_idx = int(np.argmax(scores))

        if not np.isfinite(scores[best_idx]):
            break

        support.append(best_idx)

        A_s = A[:, support]

        # Least-squares fit across all chirps.
        X, _, _, _ = np.linalg.lstsq(A_s, Y, rcond=None)

        residual = Y - A_s @ X

        residual_energy_ratio = float(np.linalg.norm(residual, "fro") ** 2) / initial_energy
        residual_history.append(residual_energy_ratio)

        if residual_energy_ratio <= residual_tol:
            break

    info = {
        "support": support,
        "residual_history": residual_history,
        "initial_energy": initial_energy,
        "final_residual_ratio": residual_history[-1] if residual_history else None,
    }

    return support, X.astype(np.complex64), info


# ============================================================
# OMP row reconstruction
# ============================================================

def build_clean_row_from_omp(
    fft_row_db: np.ndarray,
    support: Sequence[int],
    coeffs: np.ndarray,
    omp_angle_grid_rad: np.ndarray,
    display_angle_grid_rad: np.ndarray,
    clean_beam_width_bins: float = 1.4,
    residual_floor_db: Optional[float] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Convert OMP components into a display-grid row.

    The clean row contains narrow components on the display angle grid.
    """
    fft_row_db = fft_row_db.astype(np.float32)

    if residual_floor_db is None:
        residual_floor_db = float(np.percentile(fft_row_db, 10.0))

    clean_linear = np.ones_like(fft_row_db, dtype=np.float32) * db_to_linear(
        np.array(residual_floor_db, dtype=np.float32)
    )

    if len(support) == 0 or coeffs.size == 0:
        return fft_row_db.copy(), []

    component_power = np.mean(np.abs(coeffs) ** 2, axis=1).astype(np.float32)

    if np.max(component_power) <= 0:
        return fft_row_db.copy(), []

    component_db = to_db_local(component_power)

    # Align strongest OMP component to strongest FFT peak.
    fft_peak_db = float(np.max(fft_row_db))
    component_db = component_db + (fft_peak_db - float(np.max(component_db)))

    display_peak_bins = []

    x = np.arange(len(fft_row_db), dtype=np.float32)

    for i, atom_idx in enumerate(support):
        theta = omp_angle_grid_rad[int(atom_idx)]

        display_idx = int(np.argmin(np.abs(display_angle_grid_rad - theta)))
        display_peak_bins.append(display_idx)

        peak_linear = float(db_to_linear(np.array(component_db[i], dtype=np.float32)))

        kernel = np.exp(
            -0.5 * ((x - display_idx) / float(clean_beam_width_bins)) ** 2
        ).astype(np.float32)

        clean_linear += peak_linear * kernel

    clean_row_db = linear_to_db(clean_linear)

    return clean_row_db.astype(np.float32), display_peak_bins


def apply_safe_omp_update(
    fft_row_db: np.ndarray,
    clean_row_db: np.ndarray,
    init_peak_bin: int,
    clean_peak_bins: Sequence[int],
    update_mode: str = "blend_roi",
    roi_half_width_bins: int = 9,
    overlay_half_width_bins: int = 4,
    max_peak_shift_bins: int = 8,
    blend_weight: float = 0.65,
) -> Tuple[np.ndarray, bool, str]:
    """
    Safely update one FFT row with OMP-refined components.

    Modes:
        overlay:
            Only reinforces local OMP peaks.

        blend_roi:
            Replaces/blends a local region around the main target.
            This is more visibly sharpening than overlay.
    """
    fft_row_db = fft_row_db.astype(np.float32).copy()
    clean_row_db = clean_row_db.astype(np.float32)

    if len(clean_peak_bins) == 0:
        return fft_row_db, False, "no_clean_peak"

    clean_main_peak = int(clean_peak_bins[0])

    if abs(clean_main_peak - int(init_peak_bin)) > max_peak_shift_bins:
        return fft_row_db, False, "peak_shift_too_large"

    out = fft_row_db.copy()

    if update_mode == "overlay":
        for p in clean_peak_bins:
            p = int(p)
            lo = max(0, p - overlay_half_width_bins)
            hi = min(len(out), p + overlay_half_width_bins + 1)
            out[lo:hi] = np.maximum(out[lo:hi], clean_row_db[lo:hi])

    elif update_mode == "blend_roi":
        lo = max(0, int(init_peak_bin) - roi_half_width_bins)
        hi = min(len(out), int(init_peak_bin) + roi_half_width_bins + 1)

        out[lo:hi] = (
            (1.0 - blend_weight) * fft_row_db[lo:hi]
            + blend_weight * clean_row_db[lo:hi]
        )

        # Preserve peak strength so the target does not disappear.
        out[lo:hi] = np.maximum(out[lo:hi], fft_row_db[lo:hi] - 3.0)

    else:
        raise ValueError("update_mode must be 'overlay' or 'blend_roi'")

    return out.astype(np.float32), True, "accepted"


def omp_refine_one_range_bin(
    Y: np.ndarray,
    fft_row_db: np.ndarray,
    channel_positions: np.ndarray,
    display_angle_grid_rad: np.ndarray,
    omp_grid_factor: int = 4,
    max_atoms: int = 3,
    residual_tol: float = 0.08,
    d_over_lambda: float = 0.5,
    min_separation_bins: int = 4,
    clean_beam_width_bins: float = 1.4,
    update_mode: str = "blend_roi",
    roi_half_width_bins: int = 9,
    overlay_half_width_bins: int = 4,
    max_peak_shift_bins: int = 8,
    blend_weight: float = 0.65,
) -> Tuple[np.ndarray, Dict]:
    """
    Run OMP for one active range bin and safely update its FFT row.
    """
    display_size = len(display_angle_grid_rad)
    omp_grid_size = int(display_size * omp_grid_factor)

    omp_angle_grid_rad, omp_angle_grid_deg = make_fft_consistent_angle_grid(
        angle_fft_size=omp_grid_size,
        d_over_lambda=d_over_lambda,
    )

    A = steering_dictionary(
        angle_grid_rad=omp_angle_grid_rad,
        channel_positions=channel_positions,
        d_over_lambda=d_over_lambda,
        normalize=True,
    )

    init_peak_bin = int(np.argmax(fft_row_db))

    support, coeffs, omp_info = multi_snapshot_omp(
        Y=Y,
        A=A,
        max_atoms=max_atoms,
        residual_tol=residual_tol,
        min_separation_bins=min_separation_bins,
    )

    clean_row_db, clean_peak_bins = build_clean_row_from_omp(
        fft_row_db=fft_row_db,
        support=support,
        coeffs=coeffs,
        omp_angle_grid_rad=omp_angle_grid_rad,
        display_angle_grid_rad=display_angle_grid_rad,
        clean_beam_width_bins=clean_beam_width_bins,
    )

    updated_row, accepted, reason = apply_safe_omp_update(
        fft_row_db=fft_row_db,
        clean_row_db=clean_row_db,
        init_peak_bin=init_peak_bin,
        clean_peak_bins=clean_peak_bins,
        update_mode=update_mode,
        roi_half_width_bins=roi_half_width_bins,
        overlay_half_width_bins=overlay_half_width_bins,
        max_peak_shift_bins=max_peak_shift_bins,
        blend_weight=blend_weight,
    )

    info = {
        "support": support,
        "clean_peak_bins": clean_peak_bins,
        "init_peak_bin": init_peak_bin,
        "accepted": accepted,
        "reason": reason,
        "omp_grid_size": omp_grid_size,
        "omp_angle_grid_deg_selected": [
            float(omp_angle_grid_deg[s]) for s in support
        ],
        "omp_info": omp_info,
    }

    return updated_row.astype(np.float32), info


# ============================================================
# OMP image
# ============================================================

def make_omp_image_from_cube(
    V_low: np.ndarray,
    coarse_fft_image: np.ndarray,
    channel_positions: np.ndarray,
    angle_fft_size: int,
    max_atoms_per_range: int = 3,
    residual_tol: float = 0.08,
    omp_grid_factor: int = 4,
    top_range_percent: float = 18.0,
    row_max_percentile: float = 82.0,
    row_energy_percentile: float = 65.0,
    d_over_lambda: float = 0.5,
    min_separation_bins: int = 4,
    clean_beam_width_bins: float = 1.4,
    update_mode: str = "blend_roi",
    roi_half_width_bins: int = 9,
    overlay_half_width_bins: int = 4,
    max_peak_shift_bins: int = 8,
    blend_weight: float = 0.65,
) -> Tuple[np.ndarray, Dict]:
    """
    Create FFT-seeded ROI-OMP image from low-MIMO cube.

    Starts with the FFT image and updates only active range rows.
    """
    if V_low.ndim != 3:
        raise ValueError(f"V_low must be [range, chirp, channel], got {V_low.shape}")

    if coarse_fft_image.ndim != 2:
        raise ValueError(f"coarse_fft_image must be 2D, got {coarse_fft_image.shape}")

    display_angle_grid_rad, display_angle_grid_deg = make_fft_consistent_angle_grid(
        angle_fft_size=angle_fft_size,
        d_over_lambda=d_over_lambda,
    )

    active_bins = detect_active_range_bins(
        fft_image_db=coarse_fft_image,
        top_percent=top_range_percent,
        row_max_percentile=row_max_percentile,
        row_energy_percentile=row_energy_percentile,
        zero_near_range_bins=2,
    )

    omp_image = coarse_fft_image.copy().astype(np.float32)

    accepted_rows = []
    rejected_rows = {}

    range_infos = {}

    for r in active_bins:
        r = int(r)

        Y = V_low[r, :, :].T
        fft_row = coarse_fft_image[r, :]

        updated_row, info = omp_refine_one_range_bin(
            Y=Y,
            fft_row_db=fft_row,
            channel_positions=channel_positions,
            display_angle_grid_rad=display_angle_grid_rad,
            omp_grid_factor=omp_grid_factor,
            max_atoms=max_atoms_per_range,
            residual_tol=residual_tol,
            d_over_lambda=d_over_lambda,
            min_separation_bins=min_separation_bins,
            clean_beam_width_bins=clean_beam_width_bins,
            update_mode=update_mode,
            roi_half_width_bins=roi_half_width_bins,
            overlay_half_width_bins=overlay_half_width_bins,
            max_peak_shift_bins=max_peak_shift_bins,
            blend_weight=blend_weight,
        )

        if info["accepted"]:
            omp_image[r, :] = updated_row
            accepted_rows.append(r)
        else:
            rejected_rows[r] = info["reason"]

        range_infos[r] = info

    info = {
        "active_range_bins": active_bins,
        "accepted_rows": np.array(accepted_rows, dtype=np.int64),
        "rejected_rows": rejected_rows,
        "num_active_range_bins": int(len(active_bins)),
        "num_accepted_rows": int(len(accepted_rows)),
        "display_angle_grid_deg": display_angle_grid_deg,
        "range_infos": range_infos,
        "params": {
            "angle_fft_size": int(angle_fft_size),
            "max_atoms_per_range": int(max_atoms_per_range),
            "residual_tol": float(residual_tol),
            "omp_grid_factor": int(omp_grid_factor),
            "top_range_percent": float(top_range_percent),
            "row_max_percentile": float(row_max_percentile),
            "row_energy_percentile": float(row_energy_percentile),
            "d_over_lambda": float(d_over_lambda),
            "min_separation_bins": int(min_separation_bins),
            "clean_beam_width_bins": float(clean_beam_width_bins),
            "update_mode": str(update_mode),
            "roi_half_width_bins": int(roi_half_width_bins),
            "overlay_half_width_bins": int(overlay_half_width_bins),
            "max_peak_shift_bins": int(max_peak_shift_bins),
            "blend_weight": float(blend_weight),
        },
    }

    return omp_image.astype(np.float32), info


# ============================================================
# Metrics and plotting
# ============================================================

def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff ** 2))


def get_common_display_limits(
    images: Dict[str, np.ndarray],
    dynamic_range_db: float = 45.0,
) -> Tuple[float, float]:
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

    vmin, vmax = get_common_display_limits(
        images=images,
        dynamic_range_db=dynamic_range_db,
    )

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
    info_1x4: Dict,
    info_1x2: Dict,
    save_path: Path,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        f.write("Low-MIMO FFT-Seeded ROI-OMP Metrics\n")
        f.write("===================================\n\n")

        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")

        f.write("\nImprovement summary\n")
        f.write("-------------------\n")
        f.write(
            f"1x4 OMP improved MAE: "
            f"{metrics['MAE_1x8_vs_1x4_OMP'] < metrics['MAE_1x8_vs_1x4_FFT']}\n"
        )
        f.write(
            f"1x2 OMP improved MAE: "
            f"{metrics['MAE_1x8_vs_1x2_OMP'] < metrics['MAE_1x8_vs_1x2_FFT']}\n"
        )

        f.write("\nOMP 1x4 info\n")
        f.write("------------\n")
        f.write(f"Active rows: {info_1x4['num_active_range_bins']}\n")
        f.write(f"Accepted rows: {info_1x4['num_accepted_rows']}\n")
        f.write(f"Accepted row indices: {info_1x4['accepted_rows'].tolist()}\n")

        f.write("\nOMP 1x2 info\n")
        f.write("------------\n")
        f.write(f"Active rows: {info_1x2['num_active_range_bins']}\n")
        f.write(f"Accepted rows: {info_1x2['num_accepted_rows']}\n")
        f.write(f"Accepted row indices: {info_1x2['accepted_rows'].tolist()}\n")


def print_omp_info(name: str, info: Dict) -> None:
    print(f"\n{name} OMP info")
    print("-" * (len(name) + 9))
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
        description="Run Low-MIMO FFT-Seeded ROI-OMP on one real radar frame."
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

    parser.add_argument("--omp_grid_factor", type=int, default=4)
    parser.add_argument("--max_atoms_1x4", type=int, default=3)
    parser.add_argument("--max_atoms_1x2", type=int, default=1)
    parser.add_argument("--residual_tol", type=float, default=0.08)

    parser.add_argument("--top_range_percent", type=float, default=18.0)
    parser.add_argument("--row_max_percentile", type=float, default=82.0)
    parser.add_argument("--row_energy_percentile", type=float, default=65.0)

    parser.add_argument("--clean_beam_width_bins", type=float, default=1.4)
    parser.add_argument(
        "--update_mode",
        type=str,
        default="blend_roi",
        choices=["overlay", "blend_roi"],
    )
    parser.add_argument("--roi_half_width_bins", type=int, default=9)
    parser.add_argument("--overlay_half_width_bins", type=int, default=4)
    parser.add_argument("--max_peak_shift_bins", type=int, default=8)
    parser.add_argument("--blend_weight", type=float, default=0.65)

    args = parser.parse_args()

    project_root = setup_project_root(Path(args.project_root))

    import config
    import radar_processing
    from radar_io import collect_all_frames, load_adc_frame

    config.create_dirs()

    data_root = resolve_project_path(project_root, config.DATA_ROOT)

    print("\n========================================")
    print("Low-MIMO FFT-Seeded ROI-OMP")
    print("========================================")
    print(f"Project root : {project_root}")
    print(f"Dataset root : {data_root}")
    print(f"Sequence     : {args.sequence}")
    print(f"Frame        : {args.frame}")
    print(f"Angle FFT    : {config.ANGLE_FFT_SIZE}")

    print("\nOMP parameters")
    print("--------------")
    print(f"omp_grid_factor       : {args.omp_grid_factor}")
    print(f"max_atoms_1x4         : {args.max_atoms_1x4}")
    print(f"max_atoms_1x2         : {args.max_atoms_1x2}")
    print(f"residual_tol          : {args.residual_tol}")
    print(f"top_range_percent     : {args.top_range_percent}")
    print(f"row_max_percentile    : {args.row_max_percentile}")
    print(f"row_energy_percentile : {args.row_energy_percentile}")
    print(f"clean_beam_width_bins : {args.clean_beam_width_bins}")
    print(f"update_mode           : {args.update_mode}")
    print(f"roi_half_width_bins   : {args.roi_half_width_bins}")
    print(f"overlay_half_width    : {args.overlay_half_width_bins}")
    print(f"max_peak_shift_bins   : {args.max_peak_shift_bins}")
    print(f"blend_weight          : {args.blend_weight}")

    # ------------------------------------------------------------
    # 1. Load frame
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
    # 2. Natural low-MIMO configurations
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
    # 3. FFT images using original radar_processing.py
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
    # 4. Virtual cube for OMP input
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
    # 5. OMP refinement
    # ------------------------------------------------------------
    img_1x4_omp, info_1x4 = make_omp_image_from_cube(
        V_low=V_1x4,
        coarse_fft_image=img_1x4_fft,
        channel_positions=pos_1x4,
        angle_fft_size=config.ANGLE_FFT_SIZE,
        max_atoms_per_range=args.max_atoms_1x4,
        residual_tol=args.residual_tol,
        omp_grid_factor=args.omp_grid_factor,
        top_range_percent=args.top_range_percent,
        row_max_percentile=args.row_max_percentile,
        row_energy_percentile=args.row_energy_percentile,
        d_over_lambda=0.5,
        min_separation_bins=4,
        clean_beam_width_bins=args.clean_beam_width_bins,
        update_mode=args.update_mode,
        roi_half_width_bins=args.roi_half_width_bins,
        overlay_half_width_bins=args.overlay_half_width_bins,
        max_peak_shift_bins=args.max_peak_shift_bins,
        blend_weight=args.blend_weight,
    )

    img_1x2_omp, info_1x2 = make_omp_image_from_cube(
        V_low=V_1x2,
        coarse_fft_image=img_1x2_fft,
        channel_positions=pos_1x2,
        angle_fft_size=config.ANGLE_FFT_SIZE,
        max_atoms_per_range=args.max_atoms_1x2,
        residual_tol=args.residual_tol,
        omp_grid_factor=args.omp_grid_factor,
        top_range_percent=args.top_range_percent,
        row_max_percentile=args.row_max_percentile,
        row_energy_percentile=args.row_energy_percentile,
        d_over_lambda=0.5,
        min_separation_bins=4,
        clean_beam_width_bins=args.clean_beam_width_bins,
        update_mode=args.update_mode,
        roi_half_width_bins=args.roi_half_width_bins,
        overlay_half_width_bins=args.overlay_half_width_bins,
        max_peak_shift_bins=args.max_peak_shift_bins,
        blend_weight=args.blend_weight,
    )

    print_omp_info("1x4", info_1x4)
    print_omp_info("1x2", info_1x2)

    # ------------------------------------------------------------
    # 6. Metrics
    # ------------------------------------------------------------
    metrics = {
        "MAE_1x8_vs_1x4_FFT": mae(img_1x8_fft, img_1x4_fft),
        "MAE_1x8_vs_1x4_OMP": mae(img_1x8_fft, img_1x4_omp),
        "MAE_1x8_vs_1x2_FFT": mae(img_1x8_fft, img_1x2_fft),
        "MAE_1x8_vs_1x2_OMP": mae(img_1x8_fft, img_1x2_omp),
        "MSE_1x8_vs_1x4_FFT": mse(img_1x8_fft, img_1x4_fft),
        "MSE_1x8_vs_1x4_OMP": mse(img_1x8_fft, img_1x4_omp),
        "MSE_1x8_vs_1x2_FFT": mse(img_1x8_fft, img_1x2_fft),
        "MSE_1x8_vs_1x2_OMP": mse(img_1x8_fft, img_1x2_omp),
    }

    print("\nMetrics")
    print("-------")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    print(
        f"\n1x4 OMP improved MAE: "
        f"{metrics['MAE_1x8_vs_1x4_OMP'] < metrics['MAE_1x8_vs_1x4_FFT']}"
    )
    print(
        f"1x2 OMP improved MAE: "
        f"{metrics['MAE_1x8_vs_1x2_OMP'] < metrics['MAE_1x8_vs_1x2_FFT']}"
    )

    # ------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------
    out_dir = project_root / "results" / "omp_debug" / args.sequence
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = out_dir / f"{args.frame}_omp_comparison.png"
    difference_path = out_dir / f"{args.frame}_omp_difference_from_1x8.png"
    metrics_path = out_dir / f"{args.frame}_omp_metrics.txt"

    images = {
        "1x8 FFT": img_1x8_fft,
        "1x4 FFT": img_1x4_fft,
        "1x4 FFT + OMP": img_1x4_omp,
        "1x2 FFT": img_1x2_fft,
        "1x2 FFT + OMP": img_1x2_omp,
    }

    save_comparison_figure(
        images=images,
        save_path=comparison_path,
        title="Natural Low-MIMO FFT-Seeded ROI-OMP",
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
        info_1x4=info_1x4,
        info_1x2=info_1x2,
        save_path=metrics_path,
    )

    print("\nSaved")
    print("-----")
    print(f"Comparison: {comparison_path}")
    print(f"Difference: {difference_path}")
    print(f"Metrics   : {metrics_path}")


if __name__ == "__main__":
    main()