import cv2
from src.yolov8.detection import GunDetection, ClothesDetection
from src.utils import sendP, countCameras

def main(tryall=True):
    weight_path_gun = 'model/yolov8_SciFair/runs/detect/train4/weights/best.pt'
    config_path_gun = 'model/gun.cfg' #for yolov4
    gun_detection = GunDetection(weight_path_gun)

    if not tryall:
        cam1 = cv2.VideoCapture(1)
        #cam2 = cv2.VideoCapture(0)
        cameras = [
            (cam1, "Zone 1")#, (cam2, 'Zone 2')
        ]
    else:
        cameras = [(cv2.VideoCapture(i), f"Zone {i+1}") for i in range(countCameras())]

    for i in gun_detection.run_detection(cameras):
        print(type(i))
        if type(i) == dict:
            if 1 in list(i.values()):
                for key, value in i.items():
                    if value == 1:
                        print(f'Gunman in {key}')
            else:
                print('No Gunman found')
        else:
            clothes_detection = ClothesDetection(len(cameras),cameras)
            wearing,colour = clothes_detection.run_detection(cameras)
            print(f"wearing {wearing} and RGB {colour}")
            for i in gun_detection.run_detection(cameras):
                print(type(i))
                if type(i) == dict:
                    if 1 in list(i.values()):
                        for key, value in i.items():
                            if value == 1:
                                print(f'Gunman in {key}')
                    else:
                        print('No Gunman found')
    gun_detection.run_detection(cameras)                  
if __name__ == "__main__":
    main(tryall=False)
