"""Create sample OK/NG datasets for PatchCore-NG."""

from pathlib import Path

import cv2
import numpy as np


def write_good(path: Path, index: int) -> None:
    img = np.full((256, 256, 3), 180, dtype=np.uint8)
    cv2.rectangle(img, (60, 60), (196, 196), (120, 150, 110), -1)
    cv2.imwrite(str(path / f"ok_{index:02d}.png"), img)


def write_ng(path: Path, index: int) -> None:
    img = np.full((256, 256, 3), 180, dtype=np.uint8)
    cv2.rectangle(img, (60, 60), (196, 196), (120, 150, 110), -1)
    cv2.circle(img, (120 + index * 8, 80), 18, (40, 40, 220), -1)
    cv2.imwrite(str(path / f"ng_{index:02d}.png"), img)


def main() -> None:
    ok_dir = Path("data/ok")
    ng_dir = Path("data/ng")
    test_dir = Path("data/test")

    for d in [ok_dir, ng_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for i in range(10):
        write_good(ok_dir, i)
    for i in range(6):
        write_ng(ng_dir, i)

    ok = np.full((256, 256, 3), 180, dtype=np.uint8)
    cv2.rectangle(ok, (60, 60), (196, 196), (120, 150, 110), -1)
    cv2.imwrite(str(test_dir / "ok_sample.png"), ok)
    write_ng(test_dir, 99)

    print(f"OK dir : {ok_dir}  (10 images -> Memory/调参/测试 约 6:2:2)")
    print(f"NG dir : {ng_dir}  (6 images -> 调参/测试 约 5:5)")
    print(f"Test   : {test_dir}")


if __name__ == "__main__":
    main()
