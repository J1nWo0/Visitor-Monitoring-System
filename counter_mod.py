import os
import cv2
import cvzone
import time
import numpy as np
from ultralytics import YOLO
import torch
import datetime  # Import for timestamp
import pickle
import zlib

# Define a class to manage colors used in the application
class Color:
    def __init__(self):
        # Define a dictionary to store various colors used in the application
        self.colors = {
            'boundingBox1': (0, 255, 0),       # Green color for bounding box 1
            'boundingBox2': (0, 255, 255),     # Yellow color for bounding box 2
            'text1': (255, 255, 255),          # White color for primary text
            'text2': (0, 0, 0),                # Black color for secondary text
            'area1': (255, 0, 0),              # Blue color for area 1
            'area2': (0, 0, 255),              # Red color for area 2
            'point': (255, 0, 255),            # Magenta color for points
            'center_point': (255, 255, 0),     # Cyan color for center points
            'rectangle': (0, 119, 255),        # Orange color for rectangles
            'mask': (128, 128, 128)            # Gray color for masks
        }

    def __getattr__(self, item):
        return lambda: self.colors[item]
# Initialize color manager
color = Color()

# Check for CUDA device and set it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
# Load YOLO model and move it to the appropriate device
model_person = YOLO('yolo-Weights/yolo11n.pt').to(device)  # Model for person segmentation
model_face = YOLO('yolo-Weights/yolov10n-face.pt').to(device)   # Model for face detection


# Define a class to handle the counting algorithm
class Algorithm_Count:
    def __init__(self, file_path, a1, a2, frame_size):
        self.peopleEntering = dict()  
        self.entering = dict()
        self.peopleExiting = dict()
        self.exiting = dict()
        self.file_path = file_path
        self.area1 = a1
        self.area2 = a2
        self.frame_size = frame_size
        self.paused = False
        self.coordinates = []
        self.name_frame = 'People Counting System'
        self.start_time = time.time()

        # Create a named window for displaying frames
        # cv2.namedWindow(self.name_frame)

    # Method to detect objects in a frame
    def detect_BboxOnly(self, frame):
        # Detect persons and faces using different models
        results_person = model_person.track(frame, conf=0.6, classes=[0], persist=True, tracker="bytetrack.yaml")  # Detect persons only (class 0)
        results_face = model_face.track(frame, conf=0.6, classes=[0], persist=True, tracker="bytetrack.yaml")  # Detect faces

        # Process results
        person_detections = self.process_results(results_person)
        face_detections = self.process_results(results_face)

        return person_detections, face_detections

    # Method to process the detection results
    def process_results(self, results):
        detections = []
        for r in results:  # Iterate through the results
            boxes = r.boxes  # Extract bounding boxes
            masks = r.masks  # Extract segmentation masks (if present)
            for mask_id, box in enumerate(boxes):
                # Extracting bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]  # Get the top-left (x1, y1) and bottom-right (x2, y2) coordinates
                # Extracting the tracking ID (if present)
                box_id = box.id if box.id is not None else -1  # ID will be -1 if there's no ID assigned
                # Extracting confidence score
                score = box.conf[0] if box.conf is not None else 0.0  # Default to 0.0 if confidence is missing
                # Extracting the class index (if present)
                class_id = box.cls[0] if box.cls is not None else -1  # Default to -1 if class is missing
                # Extract segmentation mask if available (convert mask to numpy array)
                mask = np.array(masks[mask_id].xy, dtype=np.int32) if masks is not None else None  # Convert mask to numpy array if it exists
                # Append to the detections list as a tuple
                detections.append([int(x1), int(y1), int(x2), int(y2), int(box_id), int(class_id), float(score), mask])

        return detections
    
    # Method to display elapsed time on the frame
    def show_time(self, frame):
        elapsed_time = time.time() - self.start_time

        # Convert elapsed time to hours, minutes, seconds, and milliseconds
        milliseconds = int(elapsed_time * 1000)
        hours, milliseconds = divmod(milliseconds, 3600000)
        minutes, milliseconds = divmod(milliseconds, 60000)
        seconds = (milliseconds / 1000.0)

        # Display the time in the format "hour:minute:second.millisecond"
        time_str = "Running Time: {:02}:{:02}:{:06.3f}".format(int(hours), int(minutes), seconds)
        cvzone.putTextRect(frame, time_str, (20, 480), 1, 1, color.text1(), color.text2())

    def change_coord_point(self, x1, x2, y1, y2):
        new_x = int(x1 + (x2 - x1) * 0.5)  # 50% from the left edge
        new_y = int(y2 - (y2 - y1) * 0.04)  # 4% from the bottom edge
        return new_x, new_y

    # Method to count people entering and exiting
    def counter(self, frame, detections_person, detections_face):
        for detect in detections_person:
            x1, y1, x2, y2, box_id, class_id, score, mask = detect
            label = f"{box_id} Person: {score:.2f}"
            
            self.person_bounding_boxes(frame, x1, y1, x2, y2, box_id, class_id, score, mask)
            self.face_bounding_boxes(frame, detections_face)
            self.track_people_entering(frame, x1, y1, x2, y2, box_id, label)
            self.track_people_exiting(frame, x1, y1, x2, y2, box_id, label)
            
        self.draw_polylines(frame)

    # Method to draw bounding boxes around detected persons
    def person_bounding_boxes(self, frame, x1, y1, x2, y2, box_id, class_id, score, mask):
        # Draws bounding boxes, text, and masks on a given frame for detected persons.
        if box_id != -1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color.rectangle(), 2)
            cvzone.putTextRect(frame, f"{class_id}: {box_id}: {score:.2f}", (x1, y1 - 10), 1, 1, color.text1(), color.text2())
            cx, cy = self.change_coord_point(x1, x2, y1, y2)
            cv2.circle(frame, (cx, cy), 4, color.point(), -1)  
            # Check if mask is valid and draw it
            if mask is not None:
                # cv2.fillPoly(frame, [mask], color.mask()) # Fill the mask with a color
                cv2.polylines(frame, [mask], True, color.center_point(), 2)  # Draw the mask outline

    def face_bounding_boxes(self, frame, face_detections):
        for detect in face_detections:
            x1, y1, x2, y2, box_id, class_id, score, mask = detect
            label = f"{box_id} Face: {score:.2f}"
            if box_id != -1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color.center_point(), 1)
                # cvzone.putTextRect(frame, f"{class_id}: {box_id}: {score:.2f}", (x1, y1 - 10), 1, 1, color.text1(), color.text2())
                # Check if mask is valid and draw it
                if mask is not None:
                    # cv2.fillPoly(frame, [mask], color.mask()) # Fill the mask with a color
                    cv2.polylines(frame, [mask], True, color.center_point(), 1)  # Draw the mask outline

    # Method to track people entering a specified area
    def track_people_entering(self, frame, x1, y1, x2, y2, id, label):
        cx, cy = self.change_coord_point(x1, x2, y1, y2)
        result_p1 = cv2.pointPolygonTest(np.array(self.area2, np.int32), ((cx, cy)), False)
        if result_p1 >= 0:
            # Initialize the entry for this person if not already present
            if id not in self.peopleEntering:
                self.peopleEntering[id] = {
                    'coords': (cx, cy)
                }
            # cv2.rectangle(frame, (x1, y1), (x2, y2), color.boundingBox2(), 2)
            # cvzone.putTextRect(frame, label, (x1 + 10, y1 - 10), 1, 1, color.text1(), color.text2()) 
        if id in self.peopleEntering:
            result_p2 = cv2.pointPolygonTest(np.array(self.area1, np.int32), ((cx, cy)), False)
            if result_p2 >= 0:
                # cv2.rectangle(frame, (x1, y1), (x2, y2), color.boundingBox1(), 2)
                cv2.circle(frame, (cx, cy), 4, color.point(), -1)  
                # cvzone.putTextRect(frame, label, (x1 + 10, y1 - 10), 1, 1, color.text1(), color.text2())
                # self.entering.add(id)
                if id not in self.entering:
                    self.entering[id] = {
                        'time': datetime.datetime.now(),
                        'face_crops': None  # Initialize face_crops as None
                    }

                # Ensure the cropped face is valid before adding to the set
                if 0 <= y1 < frame.shape[0] and 0 <= y2 < frame.shape[0] and 0 <= x1 < frame.shape[1] and 0 <= x2 < frame.shape[1]:
                    if self.entering[id]['face_crops'] is None:  # Only set if not already assigned
                        face_crop = frame[y1:y2, x1:x2] # crop the face
                        serialized_frame = pickle.dumps(face_crop)  # Serialize the frame
                        compressed_frame = zlib.compress(serialized_frame)  # Compress the serialized frame
                        self.entering[id]['face_crops'] = compressed_frame  # Store the compressed crop
                # print(self.entering)

    # Method to track people exiting a specified area
    def track_people_exiting(self, frame, x1, y1, x2, y2, id, label):
        cx, cy = self.change_coord_point(x1, x2, y1, y2)
        result_p3 = cv2.pointPolygonTest(np.array(self.area1, np.int32), ((cx, cy)), False)
        if result_p3 >= 0:
            # Initialize the entry for this person if not already present
            if id not in self.peopleExiting:
                self.peopleExiting[id] = {
                    'coords': (cx, cy)
                }
            # cv2.rectangle(frame, (x1, y1), (x2, y2), color.boundingBox1(), 2)
            # cvzone.putTextRect(frame, label, (x1 + 10, y1 - 10), 1, 1, color.text1(), color.text2()) 
        if id in self.peopleExiting:
            result_p4 = cv2.pointPolygonTest(np.array(self.area2, np.int32), ((cx, cy)), False)
            if result_p4 >= 0:
                # cv2.rectangle(frame, (x1, y1), (x2, y2), color.boundingBox2(), 2)
                cv2.circle(frame, (cx, cy), 4, color.point(), -1)  
                # cvzone.putTextRect(frame, label, (x1 + 10, y1 - 10), 1, 1, color.text1(), color.text2())
                # self.exiting.add(id)
                if id not in self.exiting:
                    self.exiting[id] = {
                        'time': datetime.datetime.now(),
                        'face_crops': None  # Initialize face_crops as None
                    }

                # Ensure the cropped face is valid before adding to the set
                if 0 <= y1 < frame.shape[0] and 0 <= y2 < frame.shape[0] and 0 <= x1 < frame.shape[1] and 0 <= x2 < frame.shape[1]:
                    if self.exiting[id]['face_crops'] is None:  # Only set if not already assigned
                        face_crop = frame[y1:y2, x1:x2] # crop the face
                        serialized_frame = pickle.dumps(face_crop)  # Serialize the frame
                        compressed_frame = zlib.compress(serialized_frame)  # Compress the serialized frame
                        self.exiting[id]['face_crops'] = compressed_frame  # Store the compressed crop

    # Method to draw polylines for specified areas and display counts
    def draw_polylines(self, frame):
        cv2.polylines(frame, [np.array(self.area1, np.int32)], True, color.area1(), 2)
        cv2.polylines(frame, [np.array(self.area2, np.int32)], True, color.area2(), 2)
        enter = len(self.entering)
        exit = len(self.exiting)
        cvzone.putTextRect(frame, str(f"Enter: {enter}"), (20, 30), 1, 1, color.text1(), color.text2())
        cvzone.putTextRect(frame, str(f"Exit: {exit}"), (20, 60), 1, 1, color.text1(), color.text2())
    
    # Main method to process the video
    def main(self):
        cap = cv2.VideoCapture(self.file_path)

        # downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        # output_file_path = os.path.join(downloads_path, 'output_video.avi')
        # out = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'XVID'), 24.0, self.frame_size)

        while True:
            if not self.paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, self.frame_size)

                detections_person, detections_face = self.detect_BboxOnly(frame)
                self.counter(frame, detections_person, detections_face)

                # Return count of people entering and exiting
                result = {
                    'total_people_entering': len(self.entering),
                    'total_people_exiting': len(self.exiting),
                    'entering_details': {
                        person_id: {
                            'time': details['time'],
                            'face_crops': details['face_crops']
                        }
                        for person_id, details in self.entering.items()
                    },
                    'exiting_details': {
                        person_id: {
                            'time': details['time'],
                            'face_crops': details['face_crops']
                        }
                        for person_id, details in self.exiting.items()
                    }
                }

                # Instead of showing the frame, process it and return
                yield frame, result  # Use a generator to return frames


        cap.release()
        # out.release()
        # cv2.destroyAllWindows()

if __name__ == '__main__':
    area1 = [(359, 559), (400, 559), (667, 675), (632, 681)]
    area2 = [(346, 563), (313, 566), (579, 703), (624, 694)]
    sample_video_path = 'Sample Test File\\test_video.mp4'
    frame_width = 1280
    frame_height = int(frame_width / 16 * 9)   
    algo = Algorithm_Count(sample_video_path, area1, area2, (frame_width, frame_height))
    r = algo.main()
    print(r)