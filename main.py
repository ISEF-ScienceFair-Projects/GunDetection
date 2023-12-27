import cv2
from src.yolov3.detection import GunDetection, ClothesDetection
from src.utils import sendP, find_guy, countCameras
def main(tryall=True):
    weight_path_gun = 'model/darknetGun.weights'
    config_path_gun = 'gun.cfg'
    if not tryall:
        cam1 = cv2.VideoCapture(0)
        cam2 = cv2.VideoCapture(1)
        #cam3 = cv2.VideoCapture(2)
        cameras = [
            (cam1, "Zone 1"), (cam2, 'Zone 2')
        ]
    else:
        cameras = []
        amt = countCameras()
        for i in range(amt):
            cameras.append((cv2.VideoCapture(i), f"Zone {i+1}"))

    gun_detection = GunDetection(weight_path_gun, config_path_gun)
    find_guy(gun_detection,cameras)
    
    clothes_detection = ClothesDetection(len(cameras))
    clothes_detection.run_detection(cameras)
    next_gun_detection = GunDetection(weight_path_gun, config_path_gun)
    while True:
        find_guy(next_gun_detection,cameras)
if __name__ == "__main__":
    main()
