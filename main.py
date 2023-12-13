import cv2
from src.detection import GunDetection, ClothesDetection
#from src.arduinoBoard import sendP

def main():
    weight_path_gun = 'model/darknetGun.weights'
    config_path_gun = 'gun.cfg'
    gun_detection = GunDetection(weight_path_gun, config_path_gun)

    cam1 = cv2.VideoCapture(0)
    cameras = [
        (cam1, "cam 1"),
    ]
    statement,zone = gun_detection.run_detection(cameras)
    if statement:
        #sendP(zone)
        clothes_detection = ClothesDetection(len(cameras))
        clothes_detection.run_detection(cameras)

if __name__ == "__main__":
    main()
