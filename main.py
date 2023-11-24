import cv2
import numpy as np
from cloth.cloth_detection import *
from gun import YoloObjD, run_cameras, run
import time
import threading

def main():
    weight_path_gun = 'yolo-obj_last.weights'
    config_path_gun = 'gun.cfg'
    yolo_detector = YoloObjD(weight_path_gun, config_path_gun)

    # init
    cam1 = cv2.VideoCapture(0)
    #cam2 = cv2.VideoCapture(1)
    cameras = [(cam1, "cam 1", {"fps": 20, "width": 640, "height": 416})]
               #(cam2, "cam 2", {"fps": 20, "width": 640, "height": 416})]

    for cam, window_name, properties in cameras:
        cam.set(cv2.CAP_PROP_FPS, properties["fps"])
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, properties["width"])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, properties["height"])

    print(cv2.useOptimized())

    # gun thread
    gun_thread = threading.Thread(target=run_cameras, args=(yolo_detector, cameras))
    gun_thread.start()

    # clothes thread
    clothes_thread = threading.Thread(target=main_clothes_detection)
    clothes_thread.start()

    gun_thread.join()
    clothes_thread.join()

def main_clothes_detection():
    cap = cv2.VideoCapture(0)
    model = Load_DeepFashion2_Yolov3()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        # + to tensor
        img_tensor = tf.convert_to_tensor(frame[tf.newaxis, ...], dtype=tf.float32) / 255.0
        img_with_boxes = Draw_Bounding_Box(frame, Detect_Clothes(img_tensor, model))
        cv2.imshow("Clothes detection", img_with_boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
