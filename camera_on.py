import sys
import cv2
import cvzone
import datetime
import os
import pickle
import zlib
from queue import Queue

from PySide6.QtCore import QTimer, Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow
from main_ui import Ui_MainWindow
from counter_mod import Algorithm_Count
from set_entry import Get_Coordinates


class FrameCaptureThread(QThread):
    frame_ready = Signal(tuple)  # Signal to emit frame data
    
    def __init__(self, frame_generator):
        super().__init__()
        self.frame_generator = frame_generator
        self.running = True
        
    def run(self):
        try:
            while self.running:
                frame_data = next(self.frame_generator)
                self.frame_ready.emit(frame_data)
        except StopIteration:
            pass
            
    def stop(self):
        self.running = False
        self.wait()


class CameraFeedWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.stop_btn.setEnabled(False)

        self.file_path = 'Sample Test File\\test_video.mp4'
        self.capture_thread = None

        self.ui.start_btn.clicked.connect(self.start_feed)
        self.ui.stop_btn.clicked.connect(self.stop_feed)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.ignore_cap = False  # Ignore update_cap for first few frames

    def start_feed(self):
        self.running = False  # stop any previous loop
        if self.capture_thread:
            self.capture_thread.stop()

        self.ui.cap_1.clear()
        self.ui.cap_2.clear()
        self.ui.cap_3.clear()

        self.a1 = None
        self.a2 = None
        area = Get_Coordinates(self.file_path, (self.ui.label.width(), self.ui.label.height()))
        self.a1 = area.get_coordinates(self.a1, self.a2, 1)
        self.a2 = area.get_coordinates(self.a2, self.a1, 2)

        if self.a1 and self.a2:
            self.ignore_cap = True
            self.running = True

            # Re-create algorithm to reset memory
            self.algo = Algorithm_Count(self.file_path, self.a1, self.a2, 
                                      (self.ui.label.width(), self.ui.label.height()))
            self.algo.reset()
            self.frame_generator = self.algo.main()

            # Start fresh capture thread
            self.capture_thread = FrameCaptureThread(self.frame_generator)
            self.capture_thread.frame_ready.connect(self.handle_frame)
            self.capture_thread.start()

            self.timer.start(30)
            self.ui.start_btn.setEnabled(False)
            self.ui.stop_btn.setEnabled(True)
        else:
            print("Coordinates not set.")

    def handle_frame(self, frame_data):
        # Store the latest frame for the timer to process
        self.last_frame_data = frame_data

    def update_frame(self):
        if hasattr(self, 'last_frame_data'):
            frame, result = self.last_frame_data
            self.last_result = result  # Save for use in other functions
            self.show_face_crops(frame, self.ui.label)
            self.update_cap(result)
            self.save_crop_faces(result)

            self.ignore_cap = False

    def stop_feed(self):
        self.running = False
        self.timer.stop()

        if self.capture_thread:
            self.capture_thread.stop()
            self.capture_thread = None

        self.ui.label.setPixmap(QPixmap())
        self.ui.cap_1.setPixmap(QPixmap())
        self.ui.cap_2.setPixmap(QPixmap())
        self.ui.cap_3.setPixmap(QPixmap())

        self.ui.start_btn.setEnabled(True)
        self.ui.stop_btn.setEnabled(False)

    def save_crop_faces(self, result):
        processed_person_ids = set()
        downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        directory_name = os.path.join(downloads_path, datetime.datetime.now().strftime('%Y-%m-%d'))

        if not os.path.exists(directory_name):
            try:
                os.makedirs(directory_name)
            except Exception as e:
                print(f"Failed to create directory: {e}")
                return

        for person_id, details in result['entering_details'].items():
            if person_id in processed_person_ids:
                continue
            try:
                face_crop = pickle.loads(zlib.decompress(details['face_crops']))
                filename = os.path.join(directory_name, f"face_{details['time'].strftime('%H-%M-%S')}.jpg")
                cv2.imwrite(filename, face_crop)
                processed_person_ids.add(person_id)
            except Exception as e:
                print(f"Error saving face {person_id}: {e}")

    def show_face_crops(self, face_crops, name_label):
        face_resized = cv2.resize(face_crops, (name_label.width(), name_label.height()), interpolation=cv2.INTER_LINEAR)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_height, face_width, face_channels = face_rgb.shape
        face_bytes_per_line = face_channels * face_width
        face_qimg = QImage(face_rgb.data, face_width, face_height, face_bytes_per_line, QImage.Format_RGB888)
        face_pixmap = QPixmap.fromImage(face_qimg)
        name_label.setPixmap(face_pixmap)

    def update_cap(self, result):
        if self.ignore_cap:
            self.ignore_cap = False
            return
        temp = []
        for person_id, details in result['entering_details'].items():
            temp.insert(0, details['face_crops'])

        if temp:
            try:
                x1 = pickle.loads(zlib.decompress(temp[0]))
                self.show_face_crops(x1, self.ui.cap_1)
                if len(temp) > 1:
                    y1 = pickle.loads(zlib.decompress(temp[1]))
                    self.show_face_crops(y1, self.ui.cap_2)
                if len(temp) > 2:
                    z1 = pickle.loads(zlib.decompress(temp[2]))
                    self.show_face_crops(z1, self.ui.cap_3)
            except Exception as e:
                print(f"Error showing face crops: {e}")

    def closeEvent(self, event):
        self.stop_feed()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraFeedWindow()
    window.show()
    sys.exit(app.exec())