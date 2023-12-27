import cv2
import numpy as np
import time
from src.yolov3.yoloProcessFrame import YoloObjD, run
import numpy as np
import time
from cloth.cloth_detection import Draw_Bounding_Box, Detect_Clothes, Load_DeepFashion2_Yolov3
import tensorflow as tf

class GunDetection:
    def __init__(self, weight_path: str, config_path: str):
        self.yolo_detector = YoloObjD(weight_path, config_path)
    
    def run_detection(self, cameras: list, single_frame_mode: bool = False, single_frame_path: str = None) -> bool:
        consecutive_gun_detected_count = 0
        buffer_iteration_count = 4
        start_time = time.time()

        while True:
            combined_frames = []
            boxes_list = []
            cordlist = []
            gunManPos = {}

            for cam, window_name in cameras:
                ret, frame, boxes, cords = run(self.yolo_detector, cam, window_name, single_frame_mode, single_frame_path)
                cordlist.append(cords)
                #print(f'cam: {window_name}\ncords: {cords}')
                if cords != (0,0,0,0):
                    #print(f'Gun man is in {window_name}')
                    #yield int(window_name[5])
                    gunManPos.update({window_name: 1})

                else:
                    gunManPos.update({window_name: 0})

                cv2.putText(frame, f"{window_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if not ret:
                    continue

                combined_frames.append(frame)
                boxes_list.append(boxes)
            yield gunManPos
            '''    
            cv2.putText(frame, f"g", (int(cordlist[0][0]),int(cordlist[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"a", (0,0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"(100,0)", (100,0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"(0,100)", (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(cordlist)
            '''
            if not combined_frames:
                break
            
            combined_frames_resized = [(cv2.resize(frame, None, fx=2, fy=2), boxes) for frame, boxes in zip(combined_frames, boxes_list)]
            combined_frame = np.hstack([frame for frame, _ in combined_frames_resized])

            elapsed_time = time.time() - start_time
            fps = len(cameras) / elapsed_time
            cv2.putText(combined_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Combined Cameras", combined_frame)

            gun_detected_this_iteration = any(boxes > 0 for _, boxes in combined_frames_resized)
            print(f"{gun_detected_this_iteration}, {[boxes for _, boxes in combined_frames_resized]}")

            if gun_detected_this_iteration:
                consecutive_gun_detected_count += 1
                if consecutive_gun_detected_count == buffer_iteration_count:
                    print("Gunman detected for", buffer_iteration_count, "ticks. Clothes detection gonna fire!!")
                    for i in range(len(cordlist)):
                        if cordlist[i][0] > 0:
                            zone = i+1
                            print(f"Zone {zone}")

                    cv2.imwrite("gunImages/gunMan.jpg", combined_frame)
                    #cm.text('mms',
                    #        message='monkey spoted. engage in evasive manuvars. go go gadget go!',
                    #        file_path='gunImages/gunMan.jpg', number=2142184754, provider="T-Mobile")
                    yield True
            else:
                consecutive_gun_detected_count = 0

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        yield False

class ClothesDetection:
    def __init__(self, num_cameras: int):
        self.models = [Load_DeepFashion2_Yolov3() for _ in range(num_cameras)]
        self.cap_list = [cv2.VideoCapture(i) for i in range(num_cameras)]
        self.start_time = time.time()

    def run_detection(self, cameras: list):
        gunman_is_wearing = ""
        avg_rgb_values = []
        while len(gunman_is_wearing)==0:
            frames = [cap.read()[1] for cap in self.cap_list]
            ret_list = [frame is not None for frame in frames]
            if not all(ret_list):
                break

            img_tensors = [tf.convert_to_tensor(frame[tf.newaxis, ...], dtype=tf.float32) / 255.0 for frame in frames]
            img_with_boxes_list = []
            box_array_list = []
            
            text = ""
            
            for i, (img_tensor, model) in enumerate(zip(img_tensors, self.models)):
                img_with_boxes, box_array, text = Draw_Bounding_Box(frames[i], Detect_Clothes(img_tensor, model))

                img_with_boxes_list.append(img_with_boxes)
                box_array_list.append(box_array)

                if box_array and not gunman_is_wearing:
                    gunman_is_wearing = text
                    #cm.text('sms',
                    #        message=f'Gunman is wearing {gunman_is_wearing}', 
                    #        number=2142184754, provider="T-Mobile")
                    avg_rgb_values = self.get_avg_rgb(box_array, frames[i])
                    print(avg_rgb_values)
                    #color_category = self.get_color_category(avg_rgb_values)
                    #print(f"Color category for cam {i + 1}: {color_category}")

            elapsed_time = time.time() - self.start_time
            fps = len(cameras) / elapsed_time

            for i, img_with_boxes in enumerate(img_with_boxes_list):
                cv2.putText(img_with_boxes, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img_with_boxes, f"Gunman is wearing {gunman_is_wearing}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img_with_boxes, f"Color rgb {avg_rgb_values}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(f"Clothes detection cam {i + 1}", img_with_boxes)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        for cap in self.cap_list:
            cap.release()
        cv2.destroyAllWindows()
        return (gunman_is_wearing,avg_rgb_values)
    def get_avg_rgb(self, box_array: list, frame: np.ndarray) -> list:
        avg_rgb_values = []

        for box in box_array:
            x1, y1, x2, y2 = box
            box_region = frame[y1:y2, x1:x2]
            avg_rgb = np.mean(box_region, axis=(0, 1))
            avg_rgb_values.append(avg_rgb.tolist())

        return avg_rgb_values
    def colour_from_RGB(avg_rgb_values):
        pass

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
    
