import cv2
import numpy as np
from ultralytics import YOLO

class YoloObjD:
    def __init__(self, weight_path):
        self.classes = ["gun"]
        self.model = YOLO(weight_path)
        self.colors = [0, 0, 0]
        self.fontcolors = [240, 255, 240]

    def process_frame(self, frame: np.ndarray) -> tuple:
        img = cv2.resize(frame, None, fx=0.4, fy=0.4)
        height, width, _ = img.shape

        class_ids = []
        confidences = []
        boxes = []
        x = y = w= h = 0
        threshold = 0.9
        results = self.model.predict(frame,imgsz=640,conf=threshold)
        names = self.model.names
        for result in results:
            boxes = result.boxes.xyxy 
            for i,box in enumerate(boxes):
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]-box[0])
                h = int(box[3]-box[1])
                cv2.rectangle(img, (x, y), (x + w, y + h), self.colors, 2)
                cv2.putText(img, names[int(box[-1])], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.fontcolors, 2)
        return img, len(boxes),(x,y,w,h)

    def calculate_contrast_font_color(self, img: np.ndarray, x: int, y: int, w: int, h: int) -> list:
        contrast_color = [0, 0, 0]
        return contrast_color

def run(yolo_detector: YoloObjD, cam: cv2.VideoCapture, window_name: str, single_frame_mode: bool = False, single_frame_path: str = None) -> tuple:
    if single_frame_mode:
        frame_info = cv2.imread(single_frame_path)
    else:
        ret, frame_info = cam.read()
        if not ret:
            return False, None, 0,-1

    frame, boxes,cords = yolo_detector.process_frame(frame_info)
    print(boxes)
    return True, frame, boxes,cords
