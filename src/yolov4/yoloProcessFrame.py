import cv2
import numpy as np

class YoloObjD:
    def __init__(self, weight_path: str, config_path: str):
        self.net = cv2.dnn.readNet(weight_path, config_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        self.classes = ["gun"]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = self.net.getUnconnectedOutLayers()

        if self.output_layers.ndim == 1:
            self.output_layers = [self.layer_names[i - 1] for i in self.output_layers]
        else:
            self.output_layers = [self.layer_names[i[0] - 1] for i in self.output_layers]

        self.colors = [0, 0, 0]
        self.fontcolors = [240, 255, 240]

    def process_frame(self, frame: np.ndarray) -> tuple:
        img = cv2.resize(frame, None, fx=0.4, fy=0.4)
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00092, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []
        x = y = w= h = 0
        threshold = 0.9
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = f"{self.classes[0]} {round(confidences[i]*100)}%"
                color = self.colors[class_ids[i]]
                fontcolor = self.calculate_contrast_font_color(img, x, y, w, h)
                font_size = max(1, min(w, h) // 50)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, font_size, fontcolor, 2)

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
