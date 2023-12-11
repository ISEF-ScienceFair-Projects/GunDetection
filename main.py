import cv2
from src.detection import GunDetection, ClothesDetection

def main():
    weight_path_gun = 'yolo-obj_last.weights'
    config_path_gun = 'gun.cfg'
    gun_detection = GunDetection(weight_path_gun, config_path_gun)

    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(1)
    cam3 = cv2.VideoCapture(2)
    cameras = [
        (cam1, "cam 1"),
        (cam2, "cam 2"),
        (cam3, "cam 3")
    ]

    if gun_detection.run_detection(cameras):
        clothes_detection = ClothesDetection(len(cameras))
        clothes_detection.run_detection(cameras)

if __name__ == "__main__":
    main()
