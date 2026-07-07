"""Export PatchCore feature backbones to ONNX."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    ResNet101_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    Wide_ResNet101_2_Weights,
    Wide_ResNet50_2_Weights,
    efficientnet_b0,
    mobilenet_v3_large,
    mobilenet_v3_small,
    resnet101,
    resnet18,
    resnet50,
    wide_resnet101_2,
    wide_resnet50_2,
)


@dataclass(frozen=True)
class BackboneSpec:
    id: str
    display_name: str
    onnx_file: str
    description: str
    builder: Callable[[], nn.Module]


BACKBONE_REGISTRY: dict[str, BackboneSpec] = {}


def register(spec: BackboneSpec) -> None:
    BACKBONE_REGISTRY[spec.id] = spec


class ResNetStyleBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, target_dim: int = 1024) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        with torch.no_grad():
            feat2, feat3 = self._extract(torch.randn(1, 3, 224, 224))
        in_channels = feat2.shape[1] + feat3.shape[1]
        self.projection = nn.Conv2d(in_channels, target_dim, kernel_size=1)

    def _extract(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        feat2 = self.layer2(x)
        feat3 = self.layer3(feat2)
        feat3 = F.interpolate(feat3, size=feat2.shape[-2:], mode="bilinear", align_corners=False)
        return feat2, feat3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat2, feat3 = self._extract(x)
        return self.projection(torch.cat([feat2, feat3], dim=1))


class SequentialFeatureBackbone(nn.Module):
    def __init__(
        self,
        features: nn.Sequential,
        layer2_end: int,
        layer3_end: int,
        target_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.features = features
        self.layer2_end = layer2_end
        self.layer3_end = layer3_end
        with torch.no_grad():
            feat2, feat3 = self._extract(torch.randn(1, 3, 224, 224))
        in_channels = feat2.shape[1] + feat3.shape[1]
        self.projection = nn.Conv2d(in_channels, target_dim, kernel_size=1)

    def _forward_to(self, x: torch.Tensor, end_index: int) -> torch.Tensor:
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx == end_index:
                break
        return x

    def _extract(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat2 = self._forward_to(x, self.layer2_end)
        feat3 = self._forward_to(x, self.layer3_end)
        feat3 = F.interpolate(feat3, size=feat2.shape[-2:], mode="bilinear", align_corners=False)
        return feat2, feat3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat2, feat3 = self._extract(x)
        return self.projection(torch.cat([feat2, feat3], dim=1))


def _build_wide_resnet50() -> nn.Module:
    return ResNetStyleBackbone(wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1))


def _build_wide_resnet101() -> nn.Module:
    return ResNetStyleBackbone(wide_resnet101_2(weights=Wide_ResNet101_2_Weights.IMAGENET1K_V1))


def _build_resnet18() -> nn.Module:
    return ResNetStyleBackbone(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1))


def _build_resnet50() -> nn.Module:
    return ResNetStyleBackbone(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1))


def _build_resnet101() -> nn.Module:
    return ResNetStyleBackbone(resnet101(weights=ResNet101_Weights.IMAGENET1K_V1))


def _build_efficientnet_b0() -> nn.Module:
    net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    return SequentialFeatureBackbone(net.features, layer2_end=4, layer3_end=6)


def _build_mobilenet_v3_large() -> nn.Module:
    net = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    return SequentialFeatureBackbone(net.features, layer2_end=7, layer3_end=13)


def _build_mobilenet_v3_small() -> nn.Module:
    net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    return SequentialFeatureBackbone(net.features, layer2_end=7, layer3_end=11)


register(BackboneSpec(
    "wide_resnet50_2",
    "WideResNet-50 (推荐)",
    "wide_resnet50_2_features.onnx",
    "PatchCore 默认 backbone，精度与速度平衡",
    _build_wide_resnet50,
))
register(BackboneSpec(
    "wide_resnet101_2",
    "WideResNet-101",
    "wide_resnet101_2_features.onnx",
    "更高精度，推理较慢",
    _build_wide_resnet101,
))
register(BackboneSpec(
    "resnet50",
    "ResNet-50",
    "resnet50_features.onnx",
    "经典 backbone，工业场景常用",
    _build_resnet50,
))
register(BackboneSpec(
    "resnet101",
    "ResNet-101",
    "resnet101_features.onnx",
    "比 ResNet-50 更深，精度更高",
    _build_resnet101,
))
register(BackboneSpec(
    "resnet18",
    "ResNet-18",
    "resnet18_features.onnx",
    "轻量快速，适合边缘设备",
    _build_resnet18,
))
register(BackboneSpec(
    "efficientnet_b0",
    "EfficientNet-B0",
    "efficientnet_b0_features.onnx",
    "高效网络，速度较快",
    _build_efficientnet_b0,
))
register(BackboneSpec(
    "mobilenet_v3_large",
    "MobileNet-V3-Large",
    "mobilenet_v3_large_features.onnx",
    "移动端友好，轻量快速",
    _build_mobilenet_v3_large,
))
register(BackboneSpec(
    "mobilenet_v3_small",
    "MobileNet-V3-Small",
    "mobilenet_v3_small_features.onnx",
    "比 V3-Large 更小更快，适合极致提速",
    _build_mobilenet_v3_small,
))

# 兼容旧文件名
BACKBONE_ALIASES = {
    "wideresnet50": "wide_resnet50_2",
    "wrn50": "wide_resnet50_2",
}


def resolve_backbone_id(name: str) -> str:
    key = name.strip().lower()
    if key in BACKBONE_REGISTRY:
        return key
    if key in BACKBONE_ALIASES:
        return BACKBONE_ALIASES[key]
    raise ValueError(f"未知 backbone: {name}. 可选: {', '.join(BACKBONE_REGISTRY)}")


def export_backbone(
    backbone_id: str,
    output: Path,
    image_size: int,
    target_dim: int,
    opset: int,
) -> None:
    spec = BACKBONE_REGISTRY[resolve_backbone_id(backbone_id)]
    model = spec.builder()
    if target_dim != 1024:
        # Rebuild projection for custom target dim
        in_channels = model.projection.in_channels
        model.projection = nn.Conv2d(in_channels, target_dim, kernel_size=1)
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["input"],
        output_names=["features"],
        dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    print(f"[{spec.display_name}] -> {output}")


def export_all(models_dir: Path, image_size: int, target_dim: int, opset: int) -> None:
    for spec in BACKBONE_REGISTRY.values():
        export_backbone(spec.id, models_dir / spec.onnx_file, image_size, target_dim, opset)


def list_backbones() -> None:
    print("可用 backbone:")
    for spec in BACKBONE_REGISTRY.values():
        print(f"  {spec.id:22}  {spec.display_name:24}  {spec.onnx_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PatchCore backbones to ONNX")
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2", help="Backbone id")
    parser.add_argument("--all", action="store_true", help="Export all registered backbones")
    parser.add_argument("--list", action="store_true", help="List available backbones")
    parser.add_argument("--output", type=Path, help="Output ONNX path")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--target-dim", type=int, default=1024)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    if args.list:
        list_backbones()
        return

    if args.all:
        export_all(args.models_dir, args.image_size, args.target_dim, args.opset)
        return

    backbone_id = resolve_backbone_id(args.backbone)
    spec = BACKBONE_REGISTRY[backbone_id]
    output = args.output or args.models_dir / spec.onnx_file
    export_backbone(backbone_id, output, args.image_size, args.target_dim, args.opset)


if __name__ == "__main__":
    main()
