#!/usr/bin/env python3
"""Copy the YOLO-BEV experiment into an OpenPCDet checkout and patch registries."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys


def patch_once(path: Path, anchor: str, insertion: str) -> None:
    text = path.read_text()
    if insertion.strip() in text:
        return
    if anchor not in text:
        raise RuntimeError(f"Could not find expected anchor in {path}: {anchor!r}")
    backup = path.with_suffix(path.suffix + ".yolo_bev_backup")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text.replace(anchor, anchor + insertion, 1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("openpcdet_root", type=Path)
    args = parser.parse_args()

    root = args.openpcdet_root.expanduser().resolve()
    source = Path(__file__).resolve().parent

    required = [
        root / "pcdet/models/backbones_2d/__init__.py",
        root / "pcdet/models/dense_heads/__init__.py",
        root / "tools/cfgs/nuscenes_models",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("Not an OpenPCDet root. Missing:", *missing, sep="\n  ", file=sys.stderr)
        return 2

    copies = [
        (
            source / "pcdet/models/backbones_2d/yolo_bev_backbone.py",
            root / "pcdet/models/backbones_2d/yolo_bev_backbone.py",
        ),
        (
            source / "pcdet/models/dense_heads/yolo_3d_head.py",
            root / "pcdet/models/dense_heads/yolo_3d_head.py",
        ),
        (
            source / "tools/cfgs/nuscenes_models/yolo_bev_lidar.yaml",
            root / "tools/cfgs/nuscenes_models/yolo_bev_lidar.yaml",
        ),
    ]

    for src, dst in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied {dst.relative_to(root)}")

    backbone_registry = root / "pcdet/models/backbones_2d/__init__.py"
    patch_once(
        backbone_registry,
        "from .base_bev_backbone import BaseBEVBackbone, BaseBEVBackboneV1, BaseBEVResBackbone\n",
        "from .yolo_bev_backbone import YOLOBEVBackbone\n",
    )
    patch_once(
        backbone_registry,
        "    'BaseBEVResBackbone': BaseBEVResBackbone,\n",
        "    'YOLOBEVBackbone': YOLOBEVBackbone,\n",
    )

    head_registry = root / "pcdet/models/dense_heads/__init__.py"
    patch_once(
        head_registry,
        "from .transfusion_head import TransFusionHead\n",
        "from .yolo_3d_head import YOLO3DHead\n",
    )
    patch_once(
        head_registry,
        "    'TransFusionHead': TransFusionHead,\n",
        "    'YOLO3DHead': YOLO3DHead,\n",
    )

    print("\nRegistry patches applied.")
    print("Train from OpenPCDet/tools with:")
    print("  python train.py --cfg_file cfgs/nuscenes_models/yolo_bev_lidar.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
