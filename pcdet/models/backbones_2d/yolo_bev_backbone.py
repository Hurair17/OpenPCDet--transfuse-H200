"""YOLO-style CSP/PAN backbone for dense OpenPCDet BEV features.

Input:
    batch_dict["spatial_features"]: [B, C, H, W]

Output:
    batch_dict["spatial_features_2d"]: [B, OUT_CHANNELS, H, W]

The output resolution is kept equal to the HeightCompression resolution, so a
nuScenes voxel backbone with total stride 8 continues to use
TARGET_ASSIGNER_CONFIG.FEATURE_MAP_STRIDE = 8.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(value: float, divisor: int = 8) -> int:
    return max(divisor, int(math.ceil(value / divisor) * divisor))


def scale_depth(value: int, multiplier: float) -> int:
    return max(int(round(value * multiplier)), 1)


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        hidden = max(int(out_channels * expansion), 1)
        self.cv1 = ConvBNAct(in_channels, hidden, 1)
        self.cv2 = ConvBNAct(hidden, out_channels, 3)
        self.use_shortcut = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.use_shortcut else y


class C2f(nn.Module):
    """Compact CSP block following the split/concat pattern used by modern YOLO."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden = max(int(out_channels * expansion), 1)
        self.cv1 = ConvBNAct(in_channels, 2 * self.hidden, 1)
        self.blocks = nn.ModuleList(
            Bottleneck(self.hidden, self.hidden, shortcut=shortcut, expansion=1.0)
            for _ in range(num_blocks)
        )
        self.cv2 = ConvBNAct((2 + num_blocks) * self.hidden, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = list(self.cv1(x).chunk(2, dim=1))
        for block in self.blocks:
            parts.append(block(parts[-1]))
        return self.cv2(torch.cat(parts, dim=1))


class SPPF(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        hidden = max(in_channels // 2, 1)
        self.cv1 = ConvBNAct(in_channels, hidden, 1)
        self.cv2 = ConvBNAct(hidden * 4, out_channels, 1)
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))


class YOLOBEVBackbone(nn.Module):
    """CSP encoder + top-down FPN + bottom-up PAN for BEV tensors."""

    def __init__(self, model_cfg, input_channels: int) -> None:
        super().__init__()
        self.model_cfg = model_cfg

        width = float(model_cfg.get("WIDTH_MULTIPLE", 1.0))
        depth = float(model_cfg.get("DEPTH_MULTIPLE", 1.0))
        base_channels: Sequence[int] = model_cfg.get(
            "BASE_CHANNELS", [128, 256, 384]
        )
        if len(base_channels) != 3:
            raise ValueError("BASE_CHANNELS must contain exactly three values.")

        c3, c4, c5 = [make_divisible(float(c) * width) for c in base_channels]
        n3 = scale_depth(int(model_cfg.get("C3_BLOCKS", 3)), depth)
        n4 = scale_depth(int(model_cfg.get("C4_BLOCKS", 3)), depth)
        n5 = scale_depth(int(model_cfg.get("C5_BLOCKS", 2)), depth)
        pan_blocks = scale_depth(int(model_cfg.get("PAN_BLOCKS", 2)), depth)

        out_channels = int(model_cfg.get("OUT_CHANNELS", 256))

        # Encoder. P3 has the original HeightCompression resolution.
        self.stem = ConvBNAct(input_channels, c3, 3, 1)
        self.stage3 = C2f(c3, c3, n3)

        self.down4 = ConvBNAct(c3, c4, 3, 2)
        self.stage4 = C2f(c4, c4, n4)

        self.down5 = ConvBNAct(c4, c5, 3, 2)
        self.stage5 = nn.Sequential(C2f(c5, c5, n5), SPPF(c5, c5))

        # Top-down FPN.
        self.p5_lateral = ConvBNAct(c5, c4, 1)
        self.p4_topdown = C2f(c4 + c4, c4, pan_blocks, shortcut=False)

        self.p4_lateral = ConvBNAct(c4, c3, 1)
        self.p3_topdown = C2f(c3 + c3, c3, pan_blocks, shortcut=False)

        # Bottom-up PAN.
        self.p3_down = ConvBNAct(c3, c4, 3, 2)
        self.p4_pan = C2f(c4 + c4, c4, pan_blocks, shortcut=False)

        self.p4_down = ConvBNAct(c4, c5, 3, 2)
        self.p5_pan = C2f(c5 + c5, c5, pan_blocks, shortcut=False)

        # Aggregate all PAN scales back at P3 resolution for a single dense head.
        self.p4_reduce = ConvBNAct(c4, c3, 1)
        self.p5_reduce = ConvBNAct(c5, c3, 1)
        self.output = ConvBNAct(c3 * 3, out_channels, 3, 1)

        self.num_bev_features = out_channels

    @staticmethod
    def _upsample_like(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=reference.shape[-2:], mode="nearest")

    def forward(self, data_dict):
        x = data_dict["spatial_features"]

        p3 = self.stage3(self.stem(x))
        p4 = self.stage4(self.down4(p3))
        p5 = self.stage5(self.down5(p4))

        p5_up = self._upsample_like(self.p5_lateral(p5), p4)
        p4_td = self.p4_topdown(torch.cat((p4, p5_up), dim=1))

        p4_up = self._upsample_like(self.p4_lateral(p4_td), p3)
        p3_td = self.p3_topdown(torch.cat((p3, p4_up), dim=1))

        p4_pan = self.p4_pan(torch.cat((self.p3_down(p3_td), p4_td), dim=1))
        p5_pan = self.p5_pan(torch.cat((self.p4_down(p4_pan), p5), dim=1))

        p4_at_p3 = self._upsample_like(self.p4_reduce(p4_pan), p3_td)
        p5_at_p3 = self._upsample_like(self.p5_reduce(p5_pan), p3_td)
        output = self.output(torch.cat((p3_td, p4_at_p3, p5_at_p3), dim=1))

        data_dict["spatial_features_2d"] = output
        return data_dict
