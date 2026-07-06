"""Create sample OK/NG datasets for PatchCore-NG train and tune."""

from pathlib import Path

import cv2
import numpy as np


def write_good(path: Path, index: int) -> None:
    img = np.full((256, 256, 3), 180, dtype=np.uint8)
    cv2.rectangle(img, (60, 60), (196, 196), (120, 150, 110), -1)
    cv2.imwrite(str(path / f"good_{index:02d}.png"), img)


def main() -> None:
    train_dir = Path("data/train/good")
    tune_ok_dir = Path("data/tune/ok")
    tune_ng_dir = Path("data/tune/ng")
    test_dir = Path("data/test")

    for d in [train_dir, tune_ok_dir, tune_ng_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for i in range(5):
        write_good(train_dir, i)
    for i in range(3):
        write_good(tune_ok_dir, i)

    ng = np.full((256, 256, 3), 180, dtype=np.uint8)
    cv2.rectangle(ng, (60, 60), (196, 196), (120, 150, 110), -1)
    cv2.circle(ng, (180, 80), 18, (40, 40, 220), -1)
    cv2.imwrite(str(tune_ng_dir / "ng_01.png"), ng)
    cv2.imwrite(str(test_dir / "ng_sample.png"), ng)

    ok = np.full((256, 256, 3), 180, dtype=np.uint8)
    cv2.rectangle(ok, (60, 60), (196, 196), (120, 150, 110), -1)
    cv2.imwrite(str(test_dir / "ok_sample.png"), ok)

    print(f"Train OK : {train_dir}")
    print(f"Tune OK  : {tune_ok_dir}")
    print(f"Tune NG  : {tune_ng_dir}")
    print(f"Test     : {test_dir}")


if __name__ == "__main__":
    main()
