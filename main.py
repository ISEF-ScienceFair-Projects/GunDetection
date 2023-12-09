import cv2
import numpy as np
import time
import tensorflow as tf
from cloth.cloth_detection import Draw_Bounding_Box, Detect_Clothes, Load_DeepFashion2_Yolov3
import GatewayTexting.callMeMaybe as cm

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
                font_size = max(1, min(w, h) // 50)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, font_size, fontcolor, 2)

        return img, len(boxes)

    def calculate_contrast_font_color(self, img: np.ndarray, x: int, y: int, w: int, h: int) -> list:
        contrast_color = [0, 0, 0]
        return contrast_color

def run(yolo_detector: YoloObjD, cam: cv2.VideoCapture, window_name: str) -> tuple:
    ret, frame_info = cam.read()
    if not ret:
        return False, None, 0

    frame, boxes = yolo_detector.process_frame(frame_info)
    return True, frame, boxes


class GunDetection:
    def __init__(self, weight_path: str, config_path: str):
        self.yolo_detector = YoloObjD(weight_path, config_path)

    def run_detection(self, cameras: list) -> bool:
        consecutive_gun_detected_count = 0
        buffer_iteration_count = 4
        start_time = time.time()

        while True:
            combined_frames = []
            boxes_list = []

            for cam, window_name in cameras:
                ret, frame, boxes = run(self.yolo_detector, cam, window_name)
                if not ret:
                    continue

                combined_frames.append(frame)
                boxes_list.append(boxes)

            if not combined_frames:
                break

            combined_frames_resized = [(cv2.resize(frame, None, fx=2, fy=2), boxes) for frame, boxes in
                                       zip(combined_frames, boxes_list)]
            combined_frame = np.hstack([frame for frame, _ in combined_frames_resized])

            elapsed_time = time.time() - start_time
            fps = len(cameras) / elapsed_time
            cv2.putText(combined_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Combined Cameras", combined_frame)

            gun_detected_this_iteration = any(boxes > 0 for _, boxes in combined_frames_resized)
            print(f"{gun_detected_this_iteration}, {[boxes for _, boxes in combined_frames_resized]}")

            if gun_detected_this_iteration:
                consecutive_gun_detected_count += 1
                if consecutive_gun_detected_count >= buffer_iteration_count:
                    print("Gunman detected for", buffer_iteration_count, "ticks. Clothes detection gonna fire!!")
                    cv2.imwrite("gunImages/gunMan.jpg", combined_frame)
                    #cm.text('mms',
                    #        message='monkey spoted. engage in evasive manuvars. go go gadget go!',
                    #        file_path='gunImages/gunMan.jpg', number=2142184754, provider="T-Mobile")
                    return True
            else:
                consecutive_gun_detected_count = 0

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        return False


class ClothesDetection:
    def __init__(self, num_cameras: int):
        self.models = [Load_DeepFashion2_Yolov3() for _ in range(num_cameras)]
        self.cap_list = [cv2.VideoCapture(i) for i in range(num_cameras)]
        self.start_time = time.time()

    def run_detection(self, cameras: list):
        while True:
            frames = [cap.read()[1] for cap in self.cap_list]
            ret_list = [frame is not None for frame in frames]
            if not all(ret_list):
                break

            img_tensors = [tf.convert_to_tensor(frame[tf.newaxis, ...], dtype=tf.float32) / 255.0 for frame in frames]
            img_with_boxes_list = []
            box_array_list = []

            for i, (img_tensor, model) in enumerate(zip(img_tensors, self.models)):
                img_with_boxes, box_array = Draw_Bounding_Box(frames[i], Detect_Clothes(img_tensor, model))
                img_with_boxes_list.append(img_with_boxes)
                box_array_list.append(box_array)

                if box_array:
                    avg_rgb_values = self.get_avg_rgb(box_array, frames[i])
                    color_category = self.get_color_category(avg_rgb_values)
                    print(f"Color category for cam {i + 1}: {color_category}")

            elapsed_time = time.time() - self.start_time
            fps = len(cameras) / elapsed_time

            for i, img_with_boxes in enumerate(img_with_boxes_list):
                cv2.putText(img_with_boxes, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(f"Clothes detection cam {i + 1}", img_with_boxes)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        for cap in self.cap_list:
            cap.release()
        cv2.destroyAllWindows()

    def get_avg_rgb(self, box_array: list, frame: np.ndarray) -> list:
        avg_rgb_values = []

        for box in box_array:
            x1, y1, x2, y2 = box
            region = frame[y1:y2, x1:x2]
            avg_rgb = np.mean(region, axis=(0, 1))
            avg_rgb_values.append(avg_rgb)

        return avg_rgb_values

    def get_color_category(self, avg_rgb_values: list) -> str:
        color_thresholds = {
            'Red': (150, 30, 30),
            'White': (220, 220, 220),
            'Black': (30, 30, 30),
            'Blue': (30, 30, 150),
            'Green': (30, 150, 30)
        }

        min_distance = float('inf')
        color_category = None

        for category, threshold in color_thresholds.items():
            distance = np.linalg.norm(np.array(avg_rgb_values) - np.array(threshold))
            if distance < min_distance:
                min_distance = distance
                color_category = category

        return color_category


def main():
    weight_path_gun = 'yolo-obj_last.weights'
    config_path_gun = 'gun.cfg'
    gun_detection = GunDetection(weight_path_gun, config_path_gun)

    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(1)
    cameras = [
        (cam1, "cam 1"),
        (cam2, "cam 2")
    ]

    if gun_detection.run_detection(cameras):
        clothes_detection = ClothesDetection(len(cameras))
        clothes_detection.run_detection(cameras)


if __name__ == "__main__":
    main()
