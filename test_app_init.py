
import sys
import os
# Set PYTHONPATH
sys.path.append(r"G:\project\yolo11-chest_editted\YOLO_PROJECT")

from PyQt6.QtWidgets import QApplication
from YOLO_PROJECT.app.qt_medsam_app import Main

def test_init():
    print("Starting Main init test...")
    try:
        app = QApplication(sys.argv)
        print("QApplication created.")
        w = Main()
        print("Main window created successfully.")
        # We don't call app.exec() as we just want to test init
    except Exception as e:
        print(f"Error during Main init: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_init()
