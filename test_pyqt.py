import sys
from PyQt6.QtWidgets import QApplication, QLabel

def test():
    print("Starting PyQt6 test...")
    try:
        app = QApplication(sys.argv)
        label = QLabel("Hello PyQt6")
        label.show()
        print("Application created and shown. Entering event loop...")
        res = app.exec()
        print(f"Event loop exited with code: {res}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test()
