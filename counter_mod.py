import os
import cv2
import cvzone
import time
import numpy as np
from ultralytics import YOLO
import torch
import datetime
import pickle
import zlib
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f'Using device: {device}')

# Utility class for color management
class Color:
    def __init__(self):
        self.colors = {
            'boundingBox1': (0, 255, 0),
            'boundingBox2': (0, 255, 255),
            'text1': (255, 255, 255),
            'text2': (0, 0, 0),
            'area1': (255, 0, 0),
            'area2': (0, 0, 255),
            'point': (255, 0, 255),
            'center_point': (255, 255, 0),
            'rectangle': (0, 119, 255),
            'mask': (128, 128, 128)
        }

    def __getattr__(self, item):
        return lambda: self.colors[item]

color = Color()

# Detector wrapper class
class Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path).to(device).eval()

    def detect(self, frame):
        results = self.model.track(frame, conf=0.4, classes=[0], persist=True, tracker="bytetrack.yaml")
        detections = []
        for r in results:
            boxes, masks = r.boxes, r.masks
            for mask_id, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0]
                box_id = box.id if box.id is not None else -1
                score = box.conf[0] if box.conf is not None else 0.0
                class_id = box.cls[0] if box.cls is not None else -1
                mask = np.array(masks[mask_id].xy, dtype=np.int32) if masks is not None else None
                detections.append([int(x1), int(y1), int(x2), int(y2), int(box_id), int(class_id), float(score), mask])
        return detections

class Algorithm_Count:
    def __init__(self, file_path, a1, a2, frame_size):
        self.file_path = file_path
        self.area1 = a1
        self.area2 = a2
        self.poly_area1 = np.array(a1, np.int32)
        self.poly_area2 = np.array(a2, np.int32)
        self.frame_size = frame_size
        self.detector = Detector('yolo-Weights/yolo11n.pt')

        self.peopleEntering = {}
        self.peopleExiting = {}
        self.entering = {}
        self.exiting = {}

        self.paused = False
        self.name_frame = 'People Counting System'
        self.start_time = time.time()

    def change_coord_point(self, x1, x2, y1, y2):
        return int(x1 + (x2 - x1) * 0.5), int(y2 - (y2 - y1) * 0.04)

    def register_movement(self, area_from, area_to, tracker_dict, action_dict, x1, y1, x2, y2, id, frame):
        cx, cy = self.change_coord_point(x1, x2, y1, y2)
        if cv2.pointPolygonTest(area_from, (cx, cy), False) >= 0:
            if id not in tracker_dict:
                tracker_dict[id] = {'coords': (cx, cy)}

        if id in tracker_dict and cv2.pointPolygonTest(area_to, (cx, cy), False) >= 0:
            if id not in action_dict:
                action_dict[id] = {
                    'time': datetime.datetime.now(),
                    'face_crops': None
                }
            if action_dict[id]['face_crops'] is None and 0 <= y1 < frame.shape[0] and 0 <= y2 < frame.shape[0] and 0 <= x1 < frame.shape[1] and 0 <= x2 < frame.shape[1]:
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    action_dict[id]['face_crops'] = zlib.compress(pickle.dumps(face_crop))

    def person_bounding_boxes(self, frame, x1, y1, x2, y2, box_id, class_id, score, mask):
        if box_id != -1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color.rectangle(), 2)
            cvzone.putTextRect(frame, f"{class_id}: {box_id}: {score:.2f}", (x1, y1 - 10), 1, 1, color.text1(), color.text2())
            cx, cy = self.change_coord_point(x1, x2, y1, y2)
            cv2.circle(frame, (cx, cy), 4, color.point(), -1)
            if mask is not None:
                cv2.polylines(frame, [mask], True, color.center_point(), 2)

    def draw_polylines(self, frame):
        cv2.polylines(frame, [self.poly_area1], True, color.area1(), 2)
        cv2.polylines(frame, [self.poly_area2], True, color.area2(), 2)
        cvzone.putTextRect(frame, f"Enter: {len(self.entering)}", (20, 30), 1, 1, color.text1(), color.text2())

    def counter(self, frame, detections_person):
        for detect in detections_person:
            x1, y1, x2, y2, box_id, class_id, score, mask = detect
            self.person_bounding_boxes(frame, x1, y1, x2, y2, box_id, class_id, score, mask)
            self.register_movement(self.poly_area2, self.poly_area1, self.peopleEntering, self.entering, x1, y1, x2, y2, box_id, frame)
            self.register_movement(self.poly_area1, self.poly_area2, self.peopleExiting, self.exiting, x1, y1, x2, y2, box_id, frame)
        self.draw_polylines(frame)

    def reset(self):
        self.peopleEntering.clear()
        self.entering.clear()
        self.peopleExiting.clear()
        self.exiting.clear()
        self.start_time = time.time()

    def main(self):
        cap = cv2.VideoCapture(self.file_path)
        while True:
            if not self.paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, self.frame_size)
                detections = self.detector.detect(frame)
                self.counter(frame, detections)
                result = {
                    'total_people_entering': len(self.entering),
                    'total_people_exiting': len(self.exiting),
                    'entering_details': self.entering,
                    'exiting_details': self.exiting
                }
                yield frame, result
        cap.release()

if __name__ == '__main__':
    area1 = [(359, 559), (400, 559), (667, 675), (632, 681)]
    area2 = [(346, 563), (313, 566), (579, 703), (624, 694)]
    video_path = os.getenv("VIDEO_PATH", "Sample Test File/test_video.mp4")
    frame_size = (1280, int(1280 / 16 * 9))
    algo = Algorithm_Count(video_path, area1, area2, frame_size)
    r = algo.main()
    print(r)