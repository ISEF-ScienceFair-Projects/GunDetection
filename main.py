import cv2
from src.yolov3.detection import GunDetection, ClothesDetection
from src.utils import sendP, countCameras

def main(tryall=True):
    weight_path_gun = 'model/darknetGun.weights'
    config_path_gun = 'gun.cfg'
    gun_detection = GunDetection(weight_path_gun, config_path_gun)

    if not tryall:
        cam1 = cv2.VideoCapture(0)
        #cam2 = cv2.VideoCapture(1)
        cameras = [
            (cam1, "Zone 1")#, (cam2, 'Zone 2')
        ]
    else:
        cameras = [(cv2.VideoCapture(i), f"Zone {i+1}") for i in range(countCameras())]

    for i in gun_detection.run_detection(cameras):
        #print(f'{i}')
        print(type(i))
        if type(i) == dict:
            if 1 in list(i.values()):
                for key, value in i.items():
                    if value == 1:
                        print(f'Gunman in {key}')
                        #sendP(key[-1])
            else:
                print('No Gunman found')

            #sendP(i)
            #if type(i) != int:
            #   break
        else:
            # if this is running we have had a gun man for more that 4 ticks 
            print('you goofy goober')
            frame = cv2.imread("gunImages/gunMan.jpg")
            clothes_detection = ClothesDetection(len(cameras))
            wearing,colour = clothes_detection.run_detection(cameras)
            print(f"wearing {wearing} and RGB {colour}")

if __name__ == "__main__":
    main()
