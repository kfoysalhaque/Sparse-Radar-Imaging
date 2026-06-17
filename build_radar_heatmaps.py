# build_radar_heatmaps.py
# Save processed radar heatmap arrays as .npy files.
# These .npy files are for training.
# Visualization should load these .npy files later.

from pathlib import Path
import argparse
import sys
import json
import numpy as np


PROJECT_ROOT = Path("/mnt/ssd-nas/foysal/Sparse")


def setup_project_root(project_root: Path):
    project_root = Path(project_root)

    if not project_root.exists():
        raise FileNotFoundError(f"Project root does not exist: {project_root}")

    sys.path.insert(0, str(project_root))
    return project_root


def sample_id_from_item(item):
    seq_name = item["sequence"].name
    frame_id = item["frame_id"]
    return f"{seq_name}_{frame_id}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    project_root = setup_project_root(Path(args.project_root))

    import config
    from radar_io import collect_all_frames, load_adc_frame
    import radar_processing

    config.create_dirs()

    out_root = project_root / "Processed_Dataset" / "radar_heatmaps"
    meta_root = project_root / "Processed_Dataset" / "radar_heatmaps_meta"

    out_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    selected_configs = radar_processing.validate_configuration_names(
        ["1x8", "1x6", "1x4", "1x2"]
    )

    frames = collect_all_frames(config.DATA_ROOT)

    if args.sequence is not None:
        frames = [
            item for item in frames
            if item["sequence"].name == args.sequence
        ]

    if args.max_frames is not None:
        frames = frames[:args.max_frames]

    print(f"Selected frames: {len(frames)}")

    for i, item in enumerate(frames):
        sample_id = sample_id_from_item(item)

        expected_paths = {
            cfg: out_root / cfg / f"{sample_id}.npy"
            for cfg in selected_configs
        }

        if not args.overwrite and all(p.exists() for p in expected_paths.values()):
            print(f"[{i+1}/{len(frames)}] skipped existing {sample_id}")
            continue

        print(f"[{i+1}/{len(frames)}] processing {sample_id}")

        adc = load_adc_frame(item["radar"])

        images_by_config = {}

        for config_name in selected_configs:
            channel_indices = radar_processing.get_virtual_configuration(config_name)

            img = radar_processing.make_radar_image(
                adc,
                channel_indices=channel_indices,
                mode="clutter_removed",
                suppress_horizontal=True,
                zero_near_range_bins=2,
                use_db=True,
                normalize=False,
            )

            images_by_config[config_name] = img.astype(np.float32)

            save_dir = out_root / config_name
            save_dir.mkdir(parents=True, exist_ok=True)

            np.save(save_dir / f"{sample_id}.npy", images_by_config[config_name])

        meta = {
            "sample_id": sample_id,
            "sequence": item["sequence"].name,
            "frame_id": item["frame_id"],
            "radar_path": str(item["radar"]),
            "image_path": str(item["image"]) if item["image"] is not None else None,
            "label_path": str(item["label"]) if item["label"] is not None else None,
            "configs": list(selected_configs),
            "shape": {
                cfg: list(images_by_config[cfg].shape)
                for cfg in selected_configs
            },
            "processing": {
                "mode": "clutter_removed",
                "suppress_horizontal": True,
                "zero_near_range_bins": 2,
                "use_db": True,
                "normalize": False,
            },
        }

        with open(meta_root / f"{sample_id}.json", "w") as f:
            json.dump(meta, f, indent=2)

    print("Done.")
    print(f"Saved radar heatmaps in: {out_root}")


if __name__ == "__main__":
    main()
