#!/usr/bin/env python3
"""Build LIBERO/LIBERO-plus LeRobot metadata for DIAL/GR00T.

This script only reads LeRobot metadata/stats; it does not load the VLM or
decode videos. Use it to:
  1) inspect the metadata DIAL will generate for ``LiberoDataConfig``;
  2) merge a ``libero`` embodiment entry into a checkpoint copy so the
     checkpoint can be inspected / evaluated before fine-tuning.

Examples:
  python scripts/build_libero_metadata.py \
    --dataset-path /path/to/libero/libero_spatial \
    --output /tmp/libero_spatial_metadata.json

  python scripts/build_libero_metadata.py \
    --dataset-path /path/to/libero/libero_spatial \
    --dataset-path /path/to/libero/libero_object \
    --merge-into /path/to/checkpoint_copy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gr00t.data.dataset import (  # noqa: E402
    LeRobotSingleDatasetWithGoalImage,
    MultiEmbodimentLeRobotMixtureDataset,
)
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.experiment.data_config_dial import load_data_config  # noqa: E402


def build_metadata(
    dataset_paths: list[str],
    embodiment_tag: str,
    *,
    split: str = "[:]",
    video_backend: str = "decord",
) -> dict[str, dict]:
    tag = EmbodimentTag(embodiment_tag)
    cfg = load_data_config("libero")
    modality_configs = cfg.modality_config()

    datasets = []
    for raw_path in dataset_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"dataset path does not exist: {path}")
        # transforms=None is intentional: metadata is read from meta/*.json,
        # so we do not need the VLM processor for this utility.
        ds = LeRobotSingleDatasetWithGoalImage(
            dataset_path=path,
            modality_configs=modality_configs,
            embodiment_tag=tag,
            transforms=None,
            video_backend=video_backend,
            split=split,
        )
        datasets.append(ds)

    if len(datasets) == 1:
        metadata = datasets[0].metadata
        return {tag.value: metadata.model_dump(mode="json")}

    mixture = MultiEmbodimentLeRobotMixtureDataset(
        data_mixture=[(ds, 1.0) for ds in datasets],
        mode="train",
        balance_dataset_weights=False,
        balance_trajectory_weights=False,
        seed=42,
        metadata_config={"percentile_mixing_method": "weighted_average"},
    )
    return {
        tag_name: metadata.model_dump(mode="json")
        for tag_name, metadata in mixture.merged_metadata.items()
    }


def _merge_into_checkpoint(checkpoint_dir: Path, new_metadata: dict[str, dict]) -> Path:
    exp_cfg = checkpoint_dir / "experiment_cfg"
    exp_cfg.mkdir(parents=True, exist_ok=True)
    metadata_path = exp_cfg / "metadata.json"
    merged = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            merged = json.load(handle)
    merged.update(new_metadata)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=4)
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        action="append",
        required=True,
        help="LIBERO LeRobot dataset root (repeat for multiple suites).",
    )
    parser.add_argument(
        "--embodiment-tag",
        default="libero",
        choices=[tag.value for tag in EmbodimentTag],
        help="Embodiment tag stored in metadata.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON path (default: libero_metadata.json). Ignored with --merge-into.",
    )
    parser.add_argument(
        "--merge-into",
        default="",
        help="Checkpoint directory; merge into <ckpt>/experiment_cfg/metadata.json.",
    )
    parser.add_argument(
        "--split",
        default="[:]",
        help="Trajectory split used only to instantiate the dataset (metadata is split-independent).",
    )
    parser.add_argument(
        "--video-backend",
        default="decord",
        choices=["decord", "torchvision_av"],
    )
    args = parser.parse_args()

    metadata = build_metadata(
        args.dataset_path,
        args.embodiment_tag,
        split=args.split,
        video_backend=args.video_backend,
    )

    tag_name = args.embodiment_tag
    if tag_name not in metadata:
        raise KeyError(
            f"expected tag {tag_name!r} in generated metadata, got {sorted(metadata)}"
        )

    if args.merge_into:
        checkpoint_dir = Path(args.merge_into).expanduser().resolve()
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"checkpoint dir does not exist: {checkpoint_dir}")
        out_path = _merge_into_checkpoint(checkpoint_dir, metadata)
    else:
        out_path = Path(args.output or "libero_metadata.json").expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=4)

    entry = metadata[tag_name]
    print(f"[metadata] tag={tag_name} -> {out_path}")
    print(f"[metadata] state keys:  {sorted(entry['statistics']['state'])}")
    print(f"[metadata] action keys: {sorted(entry['statistics']['action'])}")
    print(
        "[metadata] state dims:",
        {k: v["shape"] for k, v in entry["modalities"]["state"].items()},
    )
    print(
        "[metadata] action dims:",
        {k: v["shape"] for k, v in entry["modalities"]["action"].items()},
    )


if __name__ == "__main__":
    main()
