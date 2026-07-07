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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


class FusedPreprocessBackbone(nn.Module):
    """Input: [B,3,H,W] float RGB 0~255 (H=W=image_size, resize 在 C# OpenCV). Normalize + backbone."""

    def __init__(self, backbone: nn.Module, image_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.image_size = image_size
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0
        x = (x - self.mean) / self.std
        return self.backbone(x)


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


def build_model(backbone_id: str, target_dim: int, fuse_preprocess: bool, image_size: int) -> nn.Module:
    spec = BACKBONE_REGISTRY[resolve_backbone_id(backbone_id)]
    model = spec.builder()
    if target_dim != 1024:
        in_channels = model.projection.in_channels
        model.projection = nn.Conv2d(in_channels, target_dim, kernel_size=1)
    model.eval()
    if fuse_preprocess:
        return FusedPreprocessBackbone(model, image_size)
    return model


def attach_metadata(
    output: Path,
    *,
    fuse_preprocess: bool,
    precision: str,
    image_size: int,
    target_dim: int,
) -> None:
    import onnx
    from onnx import StringStringEntryProto

    model = onnx.load(str(output))
    keys_to_remove = {
        "patchcore_preprocess",
        "patchcore_precision",
        "patchcore_image_size",
        "patchcore_target_dim",
    }
    for idx in reversed(range(len(model.metadata_props))):
        if model.metadata_props[idx].key in keys_to_remove:
            del model.metadata_props[idx]
    model.metadata_props.extend(
        [
            StringStringEntryProto(
                key="patchcore_preprocess",
                value="fused" if fuse_preprocess else "standard",
            ),
            StringStringEntryProto(key="patchcore_precision", value=precision),
            StringStringEntryProto(key="patchcore_image_size", value=str(image_size)),
            StringStringEntryProto(key="patchcore_target_dim", value=str(target_dim)),
        ]
    )
    onnx.save(model, str(output))


def simplify_onnx_model(path: Path, *, skip: bool = False) -> None:
    if skip:
        return

    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print(f"[warn] onnxsim 未安装，跳过图简化: pip install onnxsim")
        return

    model = onnx.load(str(path))
    try:
        model_simp, check = simplify(model)
    except Exception as exc:
        print(f"[warn] onnxsim 失败，保留原图 ({path.name}): {exc}")
        return

    if not check:
        print(f"[warn] onnxsim 校验未通过，保留原图: {path.name}")
        return

    onnx.save(model_simp, str(path))
    print(f"[onnxsim] simplified -> {path.name}")


def convert_to_fp16(src: Path, dst: Path) -> None:
    import onnx

    try:
        from onnxruntime.transformers.float16 import convert_float_to_float16
    except ImportError as exc:
        raise RuntimeError("FP16 转换需要 onnxruntime: pip install onnxruntime") from exc

    model = onnx.load(str(src))
    model_fp16 = convert_float_to_float16(model, keep_io_types=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_fp16, str(dst))


def quantize_int8(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)


def resolve_output_path(
    spec: BackboneSpec,
    models_dir: Path,
    output: Path | None,
    fuse_preprocess: bool,
    fp16: bool,
    int8: bool,
) -> Path:
    if output is not None:
        return output
    stem = Path(spec.onnx_file).stem
    suffix = ""
    if fuse_preprocess:
        suffix += "_fused"
    if fp16:
        suffix += "_fp16"
    if int8:
        suffix += "_int8"
    return models_dir / f"{stem}{suffix}.onnx"


def export_backbone(
    backbone_id: str,
    output: Path,
    image_size: int,
    target_dim: int,
    opset: int,
    fuse_preprocess: bool = True,
    fp16: bool = False,
    int8: bool = False,
    calibration_dir: Path | None = None,
    simplify: bool = True,
) -> None:
    spec = BACKBONE_REGISTRY[resolve_backbone_id(backbone_id)]
    model = build_model(backbone_id, target_dim, fuse_preprocess, image_size)

    if fuse_preprocess:
        dummy = torch.rand(1, 3, image_size, image_size) * 255.0
        dynamic_axes = {"input": {0: "batch"}, "features": {0: "batch"}}
    else:
        dummy = torch.randn(1, 3, image_size, image_size)
        dynamic_axes = {"input": {0: "batch"}, "features": {0: "batch"}}

    output.parent.mkdir(parents=True, exist_ok=True)
    fp32_path = output
    if fp16 or int8:
        fp32_path = output.with_name(output.stem + "._fp32tmp.onnx")

    torch.onnx.export(
        model,
        dummy,
        str(fp32_path),
        input_names=["input"],
        output_names=["features"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        dynamo=False,
    )

    simplify_onnx_model(fp32_path, skip=not simplify)

    precision = "fp32"
    attach_metadata(
        fp32_path,
        fuse_preprocess=fuse_preprocess,
        precision="fp32",
        image_size=image_size,
        target_dim=target_dim,
    )

    current = fp32_path
    if fp16:
        fp16_path = output if not int8 else output.with_name(output.stem + "._fp16tmp.onnx")
        convert_to_fp16(current, fp16_path)
        attach_metadata(
            fp16_path,
            fuse_preprocess=fuse_preprocess,
            precision="fp16",
            image_size=image_size,
            target_dim=target_dim,
        )
        if current != fp32_path:
            current.unlink(missing_ok=True)
        current = fp16_path
        precision = "fp16"

    if int8:
        quantize_int8(current, output)
        attach_metadata(
            output,
            fuse_preprocess=fuse_preprocess,
            precision="int8",
            image_size=image_size,
            target_dim=target_dim,
        )
        if current != output:
            current.unlink(missing_ok=True)
        precision = "int8"
    elif fp16 and current != output:
        import shutil

        shutil.move(str(current), str(output))

    if fp32_path.exists() and fp32_path != output and fp32_path.name.endswith("._fp32tmp.onnx"):
        fp32_path.unlink(missing_ok=True)

    print(f"[{spec.display_name}] preprocess={'fused' if fuse_preprocess else 'standard'} precision={precision} -> {output}")


def export_all(
    models_dir: Path,
    image_size: int,
    target_dim: int,
    opset: int,
    fuse_preprocess: bool,
    fp16: bool,
    int8: bool,
    calibration_dir: Path | None,
    simplify: bool,
) -> None:
    for spec in BACKBONE_REGISTRY.values():
        output = resolve_output_path(spec, models_dir, None, fuse_preprocess, fp16, int8)
        export_backbone(
            spec.id,
            output,
            image_size,
            target_dim,
            opset,
            fuse_preprocess=fuse_preprocess,
            fp16=fp16,
            int8=int8,
            calibration_dir=calibration_dir,
            simplify=simplify,
        )


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
    parser.add_argument(
        "--no-fuse-preprocess",
        action="store_true",
        help="导出旧版 ONNX（C# 侧 resize+normalize，输入为已归一化 224x224）",
    )
    parser.add_argument(
        "--no-onnxsim",
        action="store_true",
        help="跳过 onnxsim 图简化（默认 export 后自动 simplify）",
    )
    parser.add_argument("--fp16", action="store_true", help="Also export FP16 variant")
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Also export INT8 variant (dynamic quant; use scripts/quantize_backbone.py for static)",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        help="Image directory for static INT8 calibration (optional)",
    )
    args = parser.parse_args()
    fuse_preprocess = not args.no_fuse_preprocess
    simplify = not args.no_onnxsim

    if args.list:
        list_backbones()
        return

    if args.int8 and not fuse_preprocess:
        parser.error("INT8 量化需使用默认 fused ONNX，请勿加 --no-fuse-preprocess。")

    if args.all:
        export_all(
            args.models_dir,
            args.image_size,
            args.target_dim,
            args.opset,
            fuse_preprocess,
            args.fp16,
            args.int8,
            args.calibration_dir,
            simplify,
        )
        return

    backbone_id = resolve_backbone_id(args.backbone)
    spec = BACKBONE_REGISTRY[backbone_id]
    output = resolve_output_path(
        spec,
        args.models_dir,
        args.output,
        fuse_preprocess,
        args.fp16,
        args.int8,
    )
    export_backbone(
        backbone_id,
        output,
        args.image_size,
        args.target_dim,
        args.opset,
        fuse_preprocess=fuse_preprocess,
        fp16=args.fp16,
        int8=args.int8,
        calibration_dir=args.calibration_dir,
        simplify=simplify,
    )


if __name__ == "__main__":
    main()
