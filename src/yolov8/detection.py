import cv2
import numpy as np
import time
from src.yolov8.yoloProcessFrame import YoloObjD, run
import numpy as np
import time
from src.cloth.cloth_detection import Draw_Bounding_Box, Detect_Clothes, Load_DeepFashion2_Yolov3
import tensorflow as tf
import src.GatewayTexting.callMeMaybe as cm
class GunDetection:
    def __init__(self, weight_path: str):
        self.yolo_detector = YoloObjD(weight_path)
    
    def run_detection(self, cameras: list) -> bool:
        consecutive_gun_detected_count = 0
        buffer_iteration_count = 4
        start_time = time.time()

        while True:
            combined_frames = []
            boxes_list = []
            cordlist = []
            gunManPos = {}

            for cam, window_name in cameras:
                ret, frame = cam.read()
                cv2.putText(frame, f"{window_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elapsed_time = time.time() - start_time
                fps = len(cameras) / elapsed_time
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                start_time = time.time()
                if not ret:
                    continue

                combined_frames.append(frame)
            combined_frames_resized = [cv2.resize(frame, None, fx=2, fy=2) for frame in combined_frames]
            combined_frame = np.hstack([frame for frame in combined_frames_resized])
            ret, frame, boxes, cords = run(self.yolo_detector, combined_frame, window_name)
            boxes_list.append(boxes)
            cordlist.append(cords)
            if cords != (0,0,0,0):
                gunManPos.update({window_name: 1})
            else:
                gunManPos.update({window_name: 0})
            yield gunManPos
            if not combined_frames:
                break

            gun_detected_this_iteration = any(boxes > 0 for boxes in boxes_list)
            print(f"{gun_detected_this_iteration}, {[boxes for boxes in boxes_list]}")

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
    def __init__(self, num_cameras, cap_list, isframe=False, frame=None):
        self.isframe = isframe
        if not self.isframe:
            self.models = [Load_DeepFashion2_Yolov3() for _ in range(num_cameras)]
            self.cap_list = cap_list
        else:
            self.models = [Load_DeepFashion2_Yolov3()]
            self.frame = frame
        self.start_time = time.time()

    def run_detection(self, cameras: list):
        gunman_is_wearing = ""
        avg_rgb_values = []
        while len(gunman_is_wearing) == 0:
            if not self.isframe:
                frames = [cap[0].read()[1] for cap in self.cap_list]
                ret_list = [frame is not None for frame in frames]
                if not all(ret_list):
                    break
            
            if not self.isframe:
                img_tensors = [tf.convert_to_tensor(frame[tf.newaxis, ...], dtype=tf.float32) / 255.0 for frame in frames]
                img_with_boxes_list = []
                box_array_list = []
                
                text = ""
            else:
                img_tensors = [tf.convert_to_tensor(self.frame[tf.newaxis, ...], dtype=tf.float32) / 255.0]
                img_with_boxes_list = []
                box_array_list = []
                
                text = ""
            
            for i, (img_tensor, model) in enumerate(zip(img_tensors, self.models)):
                img_with_boxes, box_array, text = Draw_Bounding_Box(frames[i], Detect_Clothes(img_tensor, model))

                img_with_boxes_list.append(img_with_boxes)
                box_array_list.append(box_array)

                if box_array and not gunman_is_wearing:
                    gunman_is_wearing = text
                    cm.text('sms',
                            message=f'Gunman is wearing {gunman_is_wearing}', 
                            number=2142184754, provider="T-Mobile")
                    avg_rgb_values = self.get_avg_rgb(box_array, frames[i])
                    print(avg_rgb_values)

                    #color_category = self.get_color_category(avg_rgb_values)
                    #print(f"Color category for cam {i + 1}: {color_category}")

            elapsed_time = time.time() - self.start_time
            fps = len(cameras) / elapsed_time

            for i, img_with_boxes in enumerate(img_with_boxes_list):
                if not self.isframe:
                    cv2.putText(img_with_boxes, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img_with_boxes, f"Gunman is wearing {gunman_is_wearing}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img_with_boxes, f"Color rgb {avg_rgb_values}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow(f"Clothes detection cam {i + 1}", img_with_boxes)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if not self.isframe:
            for cap in self.cap_list:
                cap[0].release()
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
    
