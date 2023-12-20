import cv2
from src.yolov3.detection import GunDetection, ClothesDetection

def main():
    weight_path_gun = 'model/darknetGun.weights'
    config_path_gun = 'gun.cfg'
    gun_detection = GunDetection(weight_path_gun, config_path_gun)

    cam1 = cv2.VideoCapture(1)
    """cam2 = cv2.VideoCapture(1)
    cam3 = cv2.VideoCapture(2)"""
    cameras = [
        (cam1, "cam 1")
    ]

    if gun_detection.run_detection(cameras):
        clothes_detection = ClothesDetection(len(cameras))
        clothes_detection.run_detection(cameras)

if __name__ == "__main__":
    main()
