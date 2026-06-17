# visualize_saved_radar_heatmaps.py
# Load saved .npy radar heatmaps and create clean figures
# using radar_processing.py visualization functions.

from pathlib import Path
import argparse
import sys
import numpy as np


PROJECT_ROOT = Path("/media/foysal/Foysal-2/Github/Sparse-Radar-Imaging")


def setup_project_root(project_root: Path):
    project_root = Path(project_root)

    if not project_root.exists():
        raise FileNotFoundError(f"Project root does not exist: {project_root}")

    sys.path.insert(0, str(project_root))
    return project_root


def load_saved_heatmaps(radar_root: Path, sample_id: str, configs):
    images_by_config = {}

    for cfg in configs:
        path = radar_root / cfg / f"{sample_id}.npy"

        if not path.exists():
            raise FileNotFoundError(f"Missing saved heatmap: {path}")

        images_by_config[cfg] = np.load(path)

    return images_by_config


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--sequence", type=str, default="2019_04_09_cms1000")
    parser.add_argument("--frame", type=str, default="000210")
    parser.add_argument("--configs", nargs="+", default=["1x8", "1x6", "1x4", "1x2"])
    parser.add_argument("--pair", nargs=2, default=["1x8", "1x4"])
    parser.add_argument("--dynamic_range_db", type=float, default=45.0)

    args = parser.parse_args()

    project_root = setup_project_root(Path(args.project_root))

    import config
    import radar_processing

    selected_configs = radar_processing.validate_configuration_names(args.configs)
    comparison_pair = radar_processing.validate_configuration_names(args.pair)

    sample_id = f"{args.sequence}_{args.frame}"

    radar_root = project_root / "Processed_Dataset" / "radar_heatmaps"

    images_by_config = load_saved_heatmaps(
        radar_root=radar_root,
        sample_id=sample_id,
        configs=selected_configs,
    )

    out_dir = config.FIGURE_DIR / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    radar_processing.save_configuration_comparison(
        images_by_config,
        out_dir / "configuration_comparison_from_npy.png",
        dynamic_range_db=args.dynamic_range_db,
    )

    left_name, right_name = comparison_pair

    radar_processing.save_configuration_pair_comparison(
        images_by_config[left_name],
        images_by_config[right_name],
        out_dir / f"{left_name}_vs_{right_name}_from_npy.png",
        left_name=left_name,
        right_name=right_name,
        dynamic_range_db=args.dynamic_range_db,
    )

    print("Loaded saved .npy heatmaps:")
    print(f"  {sample_id}")
    print("Saved figures:")
    print(f"  {out_dir / 'configuration_comparison_from_npy.png'}")
    print(f"  {out_dir / f'{left_name}_vs_{right_name}_from_npy.png'}")


if __name__ == "__main__":
    main()