import cv2
from src.yolov8.detection import GunDetection, ClothesDetection
from src.utils import sendP, countCameras

def main(tryall=True, recursive=True):
    weight_path_gun = 'model/yolov8/5K_dataset_29epochs/runs/detect/train/weights/best.pt'
    config_path_gun = 'model/gun.cfg' #for yolov4
    gun_detection = GunDetection(weight_path_gun)

    if not tryall:
        cam1 = cv2.VideoCapture("data/testing/IMG_0864.mp4")
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
        elif recursive:
            clothes_detection = ClothesDetection(len(cameras),cameras)
            wearing,colour = clothes_detection.run_detection(cameras)
            print(f"wearing {wearing} and RGB {colour}")
            main(tryall=tryall, recursive=False)
        else:
            main(tryall=tryall, recursive=False)
    gun_detection.run_detection(cameras)                  
if __name__ == "__main__":
    main(tryall=False)
