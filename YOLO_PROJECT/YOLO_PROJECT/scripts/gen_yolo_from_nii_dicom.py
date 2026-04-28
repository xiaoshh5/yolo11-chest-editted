import os
import json
import glob
import argparse
import numpy as np
import SimpleITK as sitk
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.transform import resize as sk_resize
from skimage.measure import label as cc_label, regionprops

def ensure_dirs(out_root: str):
    img_out = os.path.join(out_root, "images")
    lbl_out = os.path.join(out_root, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    for s in ("train", "val", "test"):
        os.makedirs(os.path.join(img_out, s), exist_ok=True)
        os.makedirs(os.path.join(lbl_out, s), exist_ok=True)
    return img_out, lbl_out


def read_dicom_image(dicom_folder: str) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_folder)
    if not series_ids:
        raise RuntimeError(f"No DICOM series in {dicom_folder}")
    series_id = series_ids[0]
    file_names = reader.GetGDCMSeriesFileNames(dicom_folder, series_id)
    reader.SetFileNames(file_names)
    image = reader.Execute()
    return image


def read_nii_mask(nii_path: str) -> sitk.Image:
    return sitk.ReadImage(nii_path)


def resample_mask_to_image(mask_img: sitk.Image, ref_img: sitk.Image) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(sitk.Transform())
    resampler.SetOutputOrigin(ref_img.GetOrigin())
    resampler.SetOutputSpacing(ref_img.GetSpacing())
    resampler.SetOutputDirection(ref_img.GetDirection())
    resampler.SetSize(ref_img.GetSize())
    return resampler.Execute(mask_img)


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    mn, mx = np.min(a), np.max(a)
    if mx > mn:
        a = (a - mn) / (mx - mn)
    else:
        a = np.zeros_like(a)
    a = (a * 255.0).clip(0, 255).astype(np.uint8)
    return a


def yolo_from_bbox(xmin, ymin, xmax, ymax, img_h, img_w):
    w = xmax - xmin
    h = ymax - ymin
    cx = xmin + w / 2.0
    cy = ymin + h / 2.0
    return (cx / img_w, cy / img_h, w / img_w, h / img_h)


def components_to_yolo_lines(mask2d: np.ndarray, img_h: int, img_w: int, cls_id: int, min_area: int = 4):
    lines = []
    cc = cc_label(mask2d.astype(np.uint8), connectivity=1)
    for r in regionprops(cc):
        if r.area < min_area:
            continue
        minr, minc, maxr, maxc = r.bbox
        x_center, y_center, width, height = yolo_from_bbox(minc, minr, maxc, maxr, img_h, img_w)
        if width <= 0 or height <= 0:
            continue
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return lines


nii_files = sorted([f for f in os.listdir(LABEL_DIR) if f.endswith(".nii") or f.endswith(".nii.gz")])
case_names = [os.path.splitext(f)[0].replace(".nii", "") for f in nii_files]
if len(case_names) >= 3:
    train_cases, test_cases = train_test_split(case_names, test_size=0.1, random_state=42)
    train_cases, val_cases = train_test_split(train_cases, test_size=0.1, random_state=42)
else:
    train_cases, val_cases, test_cases = case_names, [], []


def get_split(case, train_cases, val_cases, test_cases):
    if case in test_cases:
        return "test"
    if case in val_cases:
        return "val"
    return "train"


def scan_global_classes(case_names, label_dir, case_dir):
    global_values = set()
    for case in tqdm(case_names, desc="Scan classes"):
        nii_path = None
        for ext in (".nii", ".nii.gz"):
            cand = os.path.join(label_dir, case + ext)
            if os.path.exists(cand):
                nii_path = cand
                break
        if nii_path is None:
            continue

        dicom_root = os.path.join(case_dir, case)
        subdirs = [d for d in glob.glob(os.path.join(dicom_root, "*")) if os.path.isdir(d)]
        dicom_folder = subdirs[0] if len(subdirs) == 1 else dicom_root
        try:
            dicom_img = read_dicom_image(dicom_folder)
        except Exception:
            continue
        mask_img = read_nii_mask(nii_path)
        try:
            mask_res = resample_mask_to_image(mask_img, dicom_img)
        except Exception:
            mask_arr = sitk.GetArrayFromImage(mask_img)
            dicom_sz = dicom_img.GetSize()
            target_shape = (dicom_sz[2], dicom_sz[1], dicom_sz[0])
            mask_arr_rs = sk_resize(mask_arr, target_shape, order=0, preserve_range=True, anti_aliasing=False)
            mask_arr_rs = np.rint(mask_arr_rs).astype(mask_arr.dtype)
            mask_res = sitk.GetImageFromArray(mask_arr_rs)
            mask_res.CopyInformation(dicom_img)
        mask_arr = sitk.GetArrayFromImage(mask_res)
        vals = np.unique(mask_arr)
        vals = vals[vals != 0]
        for v in vals.tolist():
            global_values.add(int(v))
    sorted_vals = sorted(global_values)
    val_to_cls = {v: i for i, v in enumerate(sorted_vals)}
    return val_to_cls, sorted_vals
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="病例根路径（包含label与DICOM子目录）")
    ap.add_argument("--label_dir", default="label", help="掩膜目录名（相对root）")
    ap.add_argument("--case_dir", required=True, help="DICOM病例目录（相对root或绝对路径）")
    ap.add_argument("--out", required=True, help="输出YOLO数据集根目录")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    args = ap.parse_args()

    root = args.root
    label_dir = os.path.join(root, args.label_dir)
    case_dir = args.case_dir if os.path.isabs(args.case_dir) else os.path.join(root, args.case_dir)
    out_root = args.out
    img_out, lbl_out = ensure_dirs(out_root)

    nii_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])
    case_names = [os.path.splitext(f)[0].replace(".nii", "") for f in nii_files]
    if len(case_names) >= 3 and (args.val_ratio > 0 or args.test_ratio > 0):
        train_cases, test_cases = train_test_split(case_names, test_size=args.test_ratio, random_state=42)
        train_cases, val_cases = train_test_split(train_cases, test_size=args.val_ratio, random_state=42)
    else:
        train_cases, val_cases, test_cases = case_names, [], []

    val_to_cls, sorted_vals = scan_global_classes(case_names, label_dir, case_dir)
    classes_json = {"mask_values": sorted_vals, "yolo_class_ids": list(range(len(sorted_vals))), "mapping": val_to_cls}
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "classes.json"), "w") as f:
        json.dump(classes_json, f, indent=2, ensure_ascii=False)

    for case in tqdm(case_names, desc="Generate YOLO dataset"):
        nii_path = None
        for ext in (".nii", ".nii.gz"):
            cand = os.path.join(label_dir, case + ext)
            if os.path.exists(cand):
                nii_path = cand
                break
        if nii_path is None:
            print(f"Skip: no mask for {case}")
            continue
        dicom_root = os.path.join(case_dir, case)
        subdirs = [d for d in glob.glob(os.path.join(dicom_root, "*")) if os.path.isdir(d)]
        dicom_folder = subdirs[0] if len(subdirs) == 1 else dicom_root
        try:
            dicom_img = read_dicom_image(dicom_folder)
        except Exception as e:
            print(f"Skip {case}: DICOM read error {e}")
            continue
        try:
            mask_img = read_nii_mask(nii_path)
            mask_res = resample_mask_to_image(mask_img, dicom_img)
        except Exception as e:
            print(f"Warn {case}: Resample error {e}, using fallback resize.")
            mask_arr = sitk.GetArrayFromImage(mask_img)
            dicom_sz = dicom_img.GetSize()
            target_shape = (dicom_sz[2], dicom_sz[1], dicom_sz[0])
            mask_arr_rs = sk_resize(mask_arr, target_shape, order=0, preserve_range=True, anti_aliasing=False)
            mask_arr_rs = np.rint(mask_arr_rs).astype(mask_arr.dtype)
            mask_res = sitk.GetImageFromArray(mask_arr_rs)
            mask_res.CopyInformation(dicom_img)
        vol_arr = sitk.GetArrayFromImage(dicom_img)
        mask_arr = sitk.GetArrayFromImage(mask_res)
        if vol_arr.shape != mask_arr.shape:
            print(f"Shape mismatch {case}: vol {vol_arr.shape} vs mask {mask_arr.shape}, skipping.")
            continue
        split = get_split(case, train_cases, val_cases, test_cases)
        img_dir = os.path.join(img_out, split)
        lbl_dir = os.path.join(lbl_out, split)
        z_slices = vol_arr.shape[0]
        for idx in range(z_slices):
            img_slice = vol_arr[idx]
            mask_slice = mask_arr[idx]
            img_h, img_w = img_slice.shape
            img_name = f"{case}_{idx:04d}.png"
            lbl_name = f"{case}_{idx:04d}.txt"
            img_path = os.path.join(img_dir, img_name)
            lbl_path = os.path.join(lbl_dir, lbl_name)
            img_uint8 = normalize_to_uint8(img_slice)
            cv2.imwrite(img_path, img_uint8)
            lines = []
            unique_vals = np.unique(mask_slice)
            unique_vals = unique_vals[unique_vals != 0]
            for v in unique_vals.tolist():
                v_int = int(v)
                if v_int not in val_to_cls:
                    new_id = len(val_to_cls)
                    val_to_cls[v_int] = new_id
                    classes_json["mapping"] = val_to_cls
                    classes_json["mask_values"] = sorted(val_to_cls.keys())
                    classes_json["yolo_class_ids"] = [val_to_cls[k] for k in sorted(val_to_cls.keys())]
                    with open(os.path.join(out_root, "classes.json"), "w") as f:
                        json.dump(classes_json, f, indent=2, ensure_ascii=False)
                cls_id = val_to_cls[v_int]
                bin_mask = (mask_slice == v_int).astype(np.uint8)
                lines.extend(components_to_yolo_lines(bin_mask, img_h, img_w, cls_id))
            if lines:
                with open(lbl_path, "w") as f:
                    f.write("\n".join(lines))
            else:
                open(lbl_path, "w").close()
    print(f"✅ YOLO dataset generated at: {out_root}")

if __name__ == "__main__":
    main()
