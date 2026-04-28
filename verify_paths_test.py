from pathlib import Path
import sys

# Mimic the logic in qt_medsam_app.py
# We assume this script is running from the PROJECT ROOT for simplicity in this test, 
# or we just hardcode the root to where we know it is for the test.
PROJECT_ROOT = Path(r"g:\project\yolo11-chest_editted")

print(f"Testing path discovery from: {PROJECT_ROOT}")

# Possible locations for weights
search_roots = [
    PROJECT_ROOT / "runs",
    PROJECT_ROOT / "YOLO_PROJECT" / "YOLO_PROJECT" / "runs", # nested
    PROJECT_ROOT # Just root
]

print("Scanning for YOLO weights...")
yolo_files = []
for root in search_roots:
    if root.exists():
        yolo_files.extend(list(root.glob("**/weights/best.pt")))

yolo_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
seen = set()
unique_yolo = []
for f in yolo_files:
    if str(f) not in seen:
        seen.add(str(f))
        unique_yolo.append(f)

print(f"Found {len(unique_yolo)} unique YOLO weights:")
for f in unique_yolo:
    print(f" - {f}")

print("\nScanning for MedSAM weights...")
medsam_files = []
for root in search_roots:
    if root.exists():
        medsam_files.extend(list(root.glob("**/weights/best.pth"))) 
        medsam_files.extend(list(root.glob("**/sam_b.pt")))      

medsam_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
seen = set()
unique_medsam = []
for f in medsam_files:
    if str(f) not in seen:
        seen.add(str(f))
        unique_medsam.append(f)

print(f"Found {len(unique_medsam)} unique MedSAM weights:")
for f in unique_medsam:
    print(f" - {f}")
