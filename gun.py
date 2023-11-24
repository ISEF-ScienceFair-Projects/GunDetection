import cv2
import numpy as np

class YoloObjD:
    def __init__(self, weight_path, config_path):
        self.net = cv2.dnn.readNet(weight_path, config_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        self.classes = ["Gun"]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = self.net.getUnconnectedOutLayers()

        if self.output_layers.ndim == 1:
            self.output_layers = [self.layer_names[i - 1] for i in self.output_layers]
        else:
            self.output_layers = [self.layer_names[i[0] - 1] for i in self.output_layers]

        self.colors = [0, 0, 0]

    def process_frame(self, frame):
        img = cv2.resize(frame, None, fx=0.4, fy=0.4)
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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
                label = str(self.classes[class_ids[i]])
                color = self.colors[class_ids[i]]
                font_size = max(1, min(w, h) // 10) #mathsss
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, font_size, color, 2)

        return img, len(boxes)

def run_cameras(obj,cameras):
    while True:
        combined_frames = []

        for cam, window_name, properties in cameras:
            ret, frame, boxes = run(obj,cam, window_name)
            if not ret:
                continue

            combined_frames.append((frame, boxes))

        if not combined_frames:
            break

        combined_frames_resized = [(cv2.resize(frame, None, fx=2, fy=2), boxes) for frame, boxes in combined_frames]
        combined_frame = np.hstack([frame for frame, _ in combined_frames_resized])
        cv2.imshow("Combined Cameras", combined_frame)

        for _, boxes in combined_frames_resized:
            if boxes > 0:
                cv2.imwrite(f'gunImages/combined_frames.jpg', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cam, _ in cameras:
        cam.release()
    cv2.destroyAllWindows()

def main():
    weight_path = 'yolo-obj_last.weights'
    config_path = 'gun.cfg'
    yolo_detector = YoloObjD(weight_path, config_path)

    # init
    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(1)
    cameras = [(cam1, "cam 1", {"fps": 20, "width": 640, "height": 416}),
            (cam2, "cam 2", {"fps": 20, "width": 640, "height": 416})]

    for cam, window_name, properties in cameras:
        cam.set(cv2.CAP_PROP_FPS, properties["fps"])
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, properties["width"])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, properties["height"])


    print(cv2.useOptimized())
    run_cameras(yolo_detector,cameras)
    
def run(yolo_detector,cam, window_name):
    ret, frame_info = cam.read()
    if not ret:
        return False, None, 0

    frame, boxes = yolo_detector.process_frame(frame_info)
    return True, frame, boxes
if __name__ == "__main__":
    main()

