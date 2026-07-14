"""YOLO-style decoupled dense 3D head for OpenPCDet.

This class deliberately subclasses CenterHead so the first experiment reuses
OpenPCDet's tested:
  * nuScenes target encoding
  * Gaussian center assignment
  * 3D box and velocity decoding
  * rotated NMS
  * final_box_dicts interface

The changed part is the prediction architecture: one all-class, decoupled
classification/regression head with C2f towers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .center_head import CenterHead
from ..backbones_2d.yolo_bev_backbone import C2f, ConvBNAct


class YOLODecoupledPredictionHead(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        separate_head_cfg,
        tower_depth: int = 2,
    ) -> None:
        super().__init__()
        self.head_order = list(separate_head_cfg.HEAD_ORDER)

        self.cls_tower = nn.Sequential(
            C2f(input_channels, input_channels, tower_depth, shortcut=False),
            ConvBNAct(input_channels, input_channels, 3),
        )
        self.reg_tower = nn.Sequential(
            C2f(input_channels, input_channels, tower_depth, shortcut=False),
            ConvBNAct(input_channels, input_channels, 3),
        )

        self.hm = nn.Conv2d(input_channels, num_classes, kernel_size=1, bias=True)
        nn.init.constant_(self.hm.bias, -2.19)

        self.reg_heads = nn.ModuleDict()
        for name in self.head_order:
            if name not in separate_head_cfg.HEAD_DICT:
                raise KeyError(f"Missing HEAD_DICT entry for {name!r}")
            out_channels = int(separate_head_cfg.HEAD_DICT[name]["out_channels"])
            layer = nn.Sequential(
                ConvBNAct(input_channels, input_channels, 3),
                nn.Conv2d(input_channels, out_channels, kernel_size=1, bias=True),
            )
            nn.init.normal_(layer[-1].weight, mean=0.0, std=0.001)
            nn.init.constant_(layer[-1].bias, 0.0)
            self.reg_heads[name] = layer

    def forward(self, x: torch.Tensor):
        cls_features = self.cls_tower(x)
        reg_features = self.reg_tower(x)

        output = {"hm": self.hm(cls_features)}
        for name in self.head_order:
            output[name] = self.reg_heads[name](reg_features)
        return output


class YOLO3DHead(CenterHead):
    """Single-group YOLO-style prediction towers with CenterHead geometry."""

    def __init__(
        self,
        model_cfg,
        input_channels,
        num_class,
        class_names,
        grid_size,
        point_cloud_range,
        voxel_size,
        predict_boxes_when_training=True,
    ) -> None:
        # Build the standard target/loss/decoder machinery first.
        super().__init__(
            model_cfg=model_cfg,
            input_channels=input_channels,
            num_class=num_class,
            class_names=class_names,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            predict_boxes_when_training=predict_boxes_when_training,
        )

        if len(self.class_names_each_head) != 1:
            raise ValueError(
                "YOLO3DHead requires one CLASS_NAMES_EACH_HEAD group containing "
                "all classes, because YOLO predicts all classes in one dense head."
            )

        head_channels = int(
            model_cfg.get("HEAD_CHANNELS", model_cfg.SHARED_CONV_CHANNEL)
        )
        tower_depth = int(model_cfg.get("TOWER_DEPTH", 2))

        # Replace CenterHead's shared conv and SeparateHead modules.
        self.shared_conv = nn.Sequential(
            ConvBNAct(input_channels, head_channels, 3),
            C2f(head_channels, head_channels, tower_depth, shortcut=True),
        )
        self.heads_list = nn.ModuleList(
            [
                YOLODecoupledPredictionHead(
                    input_channels=head_channels,
                    num_classes=len(self.class_names_each_head[0]),
                    separate_head_cfg=self.separate_head_cfg,
                    tower_depth=tower_depth,
                )
            ]
        )
