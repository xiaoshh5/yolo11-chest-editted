import sys
from PyQt6.QtWidgets import QApplication
from YOLO_PROJECT.app.qt_medsam_app import Main
import time
import threading

def test_app_launch():
    print("Testing PyQt app launch...", flush=True)
    app = QApplication(sys.argv)
    window = Main()
    window.show()
    print("App window shown.", flush=True)
    
    # Close after 5 seconds to verify it launched okay
    def close_app():
        time.sleep(5)
        print("Closing app...", flush=True)
        app.quit()
        
    threading.Thread(target=close_app, daemon=True).start()
    sys.exit(app.exec())

if __name__ == "__main__":
    test_app_launch()
