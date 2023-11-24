import cv2
import numpy as np
import time
from cloth.cloth_detection import Draw_Bounding_Box, Detect_Clothes, Load_DeepFashion2_Yolov3
from gun import YoloObjD, run
import tensorflow as tf

def main():
    weight_path_gun = 'yolo-obj_last.weights'
    config_path_gun = 'gun.cfg'
    yolo_detector = YoloObjD(weight_path_gun, config_path_gun)

    # init
    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(1)
    cameras = [
        (cam1, "cam 1"),
        (cam2, "cam 2")
    ]

    run_gun_detection(yolo_detector, cameras)

def run_gun_detection(yolo_detector, cameras):
    consecutive_gun_detected_count = 0
    buffer_iteration_count = 4
    start_time = time.time()

    while True:
        combined_frames = []

        for cam, window_name in cameras:
            ret, frame, boxes = run(yolo_detector, cam, window_name)
            if not ret:
                continue

            combined_frames.append((frame, boxes))

        if not combined_frames:
            break

        combined_frames_resized = [(cv2.resize(frame, None, fx=2, fy=2), boxes) for frame, boxes in combined_frames]
        combined_frame = np.hstack([frame for frame, _ in combined_frames_resized])

        # FPS maths
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
                run_clothes_detection(cameras)
                return
        else:
            consecutive_gun_detected_count = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cam, _ in cameras:
        cam.release()
    cv2.destroyAllWindows()

def run_clothes_detection(cameras):
    models = [Load_DeepFashion2_Yolov3() for _ in range(len(cameras))]
    cap_list = [cv2.VideoCapture(i) for i in range(len(cameras))]
    start_time = time.time()
    
    while True:
        
        for i, cap in enumerate(cap_list):
            ret, frame = cap.read()
            if not ret:
                print(f"falure on cam{i}")
                break
            

            img_tensor = tf.convert_to_tensor(frame[tf.newaxis, ...], dtype=tf.float32) / 255.0
            img_with_boxes = Draw_Bounding_Box(frame, Detect_Clothes(img_tensor, models[i]))
            elapsed_time = time.time() - start_time
            fps = len(cameras) / elapsed_time
            cv2.putText(img_with_boxes, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(f"Clothes detection cam {i+1}", img_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in cap_list:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
