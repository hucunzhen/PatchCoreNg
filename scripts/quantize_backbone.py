"""Post-export INT8 quantization for PatchCore backbone ONNX models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_calibration_reader(
    model_path: Path,
    data_dir: Path,
    image_size: int,
    fuse_preprocess: bool,
    max_samples: int = 100,
):
    import cv2
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_paths = [
        p for p in sorted(data_dir.rglob("*")) if p.suffix.lower() in extensions
    ][:max_samples]

    if not image_paths:
        raise FileNotFoundError(f"校准目录无图像: {data_dir}")

    feeds: list[dict[str, np.ndarray]] = []
    for path in image_paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if fuse_preprocess:
            rgb = rgb.astype(np.float32)
            chw = np.transpose(rgb, (2, 0, 1))
            feeds.append({input_name: np.expand_dims(chw, axis=0)})
        else:
            blob = cv2.dnn.blobFromImage(
                rgb,
                scalefactor=1.0 / 255.0,
                size=(image_size, image_size),
                mean=(0.485, 0.456, 0.406),
                swapRB=False,
                crop=False,
            )
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
            blob /= std
            feeds.append({input_name: blob})

    if not feeds:
        raise RuntimeError("无法从校准目录读取有效图像。")

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._iter = iter(feeds)

        def get_next(self) -> dict[str, np.ndarray] | None:
            return next(self._iter, None)

    return Reader()


def quantize(
    input_path: Path,
    output_path: Path,
    calibration_dir: Path | None,
    image_size: int,
    fuse_preprocess: bool,
    static: bool,
) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic, quantize_static

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if static:
        if calibration_dir is None:
            raise ValueError("静态 INT8 量化需要 --calibration-dir")
        reader = build_calibration_reader(
            input_path,
            calibration_dir,
            image_size=image_size,
            fuse_preprocess=fuse_preprocess,
        )
        quantize_static(
            str(input_path),
            str(output_path),
            reader,
            weight_type=QuantType.QUInt8,
        )
    else:
        quantize_dynamic(
            str(input_path),
            str(output_path),
            weight_type=QuantType.QUInt8,
        )

    import onnx
    from onnx import StringStringEntryProto

    model = onnx.load(str(output_path))
    for idx in reversed(range(len(model.metadata_props))):
        if model.metadata_props[idx].key == "patchcore_precision":
            del model.metadata_props[idx]
    model.metadata_props.append(StringStringEntryProto(key="patchcore_precision", value="int8"))
    onnx.save(model, str(output_path))
    print(f"INT8 -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize PatchCore backbone ONNX to INT8")
    parser.add_argument("--input", type=Path, required=True, help="Source ONNX (fp32/fp16 fused recommended)")
    parser.add_argument("--output", type=Path, help="Output INT8 ONNX path")
    parser.add_argument("--calibration-dir", type=Path, help="OK sample images for static quantization")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--fuse-preprocess",
        action="store_true",
        help="Input model expects RGB 0~255 CHW (fused preprocess)",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Use static quantization with calibration data",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        stem = args.input.stem
        if not stem.endswith("_int8"):
            stem = f"{stem}_int8"
        output = args.input.with_name(stem + args.input.suffix)

    quantize(
        args.input,
        output,
        args.calibration_dir,
        args.image_size,
        args.fuse_preprocess,
        args.static,
    )


if __name__ == "__main__":
    main()
