"""Generate synthetic test data for the pipeline.

Creates:
  - tests/data/dicom_sample/  — 5 synthetic CT slices (256×256)
  - tests/data/sample_dataset/ — YOLO-compatible images and labels
"""
from pathlib import Path
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def make_dicom_sample():
    """Generate 5 synthetic DICOM CT slices using SimpleITK (more robust)."""
    try:
        import SimpleITK as sitk
    except ImportError:
        print("SimpleITK not installed, skipping DICOM generation")
        return

    dicom_dir = DATA / "dicom_sample"
    dicom_dir.mkdir(parents=True, exist_ok=True)

    for i in range(5):
        hu = np.zeros((256, 256), dtype=np.int16) - 1000
        rr, cc = np.meshgrid(np.arange(256), np.arange(256))

        left_lung = ((rr - 96) / 56) ** 2 + ((cc - 128) / 80) ** 2 < 1
        right_lung = ((rr - 160) / 56) ** 2 + ((cc - 128) / 80) ** 2 < 1
        hu[left_lung] = -700
        hu[right_lung] = -700

        nodule = ((rr - 96) / 6) ** 2 + ((cc - 100 + i * 2) / 6) ** 2 < 1
        ggo = ((rr - 96) / 10) ** 2 + ((cc - 100 + i * 2) / 10) ** 2 < 1
        hu[ggo] = -400
        hu[nodule] = 100

        img = sitk.GetImageFromArray(hu)
        img.SetSpacing([1.0, 1.0, 1.0])
        img.SetOrigin([0, 0, i])
        # Set consistent series metadata so ImageSeriesReader finds them
        for key in [
            "0008|103e",  # SeriesDescription
            "0020|000e",  # SeriesInstanceUID
        ]:
            img.SetMetaData(key, "1.2.3.4.5.6.7.8.9.0")
        img.SetMetaData("0020|0013", str(i + 1))  # InstanceNumber
        fname = str(dicom_dir / f"slice_{i:04d}.dcm")
        sitk.WriteImage(img, fname)

    print(f"Created {5} synthetic DICOM slices in {dicom_dir}")


def make_sample_dataset():
    """Generate a tiny YOLO-compatible dataset: 6 images with 1 nodule each."""
    img_dir = DATA / "sample_dataset" / "images"
    lbl_dir = DATA / "sample_dataset" / "labels"
    for d in [img_dir, lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for i in range(6):
        img = np.zeros((256, 256, 3), dtype=np.uint8) + 40
        cv2.ellipse(img, (128, 128), (80, 110), 0, 0, 360, (60, 60, 60), -1)
        cx, cy = 100 + i * 10, 100 + i * 3
        cv2.circle(img, (cx, cy), 8, (200, 200, 200), -1)

        h, w = img.shape[:2]
        x, y, bw, bh = cx / w, cy / h, 18.0 / w, 18.0 / h
        cv2.imwrite(str(img_dir / f"sample_{i:03d}.jpg"), img)
        (lbl_dir / f"sample_{i:03d}.txt").write_text(
            f"0 {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n"
        )

    (DATA / "sample_dataset" / "dataset.yaml").write_text(
        f"path: {DATA.as_posix()}/sample_dataset\n"
        "train: images\n"
        "val: images\n"
        "nc: 1\n"
        "names: ['lung_nodule']\n"
    )
    print(f"Created sample dataset: {len(list(img_dir.iterdir()))} images")


if __name__ == "__main__":
    make_dicom_sample()
    make_sample_dataset()
