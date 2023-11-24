import cv2
import numpy as np

class YoloObjD:
    def __init__(self, weight_path, config_path):
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

    def process_frame(self, frame):
        img = cv2.resize(frame, None, fx=0.4, fy=0.4)
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00092, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.9:
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
                font_size = max(1, min(w, h) // 50)  # mathsss
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, font_size, fontcolor, 2)

        return img, len(boxes)

    def calculate_contrast_font_color(self, img, x, y, w, h):
        roi = img[y:y+h, x:x+w]
        mean_lab = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2LAB), axis=(0, 1))
        lightness = mean_lab[0]
        contrast_color = [0, 0, 0] if lightness < 50 else [255, 255, 255]
        return contrast_color

def run(yolo_detector, cam, window_name):
    ret, frame_info = cam.read()
    if not ret:
        return False, None, 0

    frame, boxes = yolo_detector.process_frame(frame_info)
    return True, frame, boxes