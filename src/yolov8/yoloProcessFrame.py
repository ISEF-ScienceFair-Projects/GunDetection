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
        threshold = 0.5
        results = self.model(frame, show=True, conf=threshold, device="mps")
        names = self.model.names
        for result in results:
            boxes = result.boxes.xyxy 
            for i,box in enumerate(boxes):
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]-box[0])
                h = int(box[3]-box[1])
                #cv2.rectangle(img, (x, y), (x + w, y + h), self.colors, 2)
                #cv2.putText(img, names[int(box[-1])], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.fontcolors, 2)
        return img, len(boxes),(x,y,w,h)

    def calculate_contrast_font_color(self, img: np.ndarray, x: int, y: int, w: int, h: int) -> list:
        contrast_color = [0, 0, 0]
        return contrast_color

def run(yolo_detector: YoloObjD, frame: np.ndarray, window_name: str) -> tuple:
    frame_info, boxes,cords = yolo_detector.process_frame(frame)
    print(boxes)
    return True, frame_info, boxes,cords
