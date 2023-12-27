from typing import List, Dict
import serial
import cv2

def find_guy(gun_detection,cameras):
        for i in gun_detection.run_detection(cameras):
            if 1 in list(i.values()):
                for key, value in i.items():
                    if value == 1:
                        print(f'Gunman in {key}')
            else:
                print('No Gunman found')

            #sendP(i)
            #if type(i) != int:
            #   break
def countCameras(maxCam: int = 10) -> int:
    n = 0
    for i in range(maxCam):
        try:
            cap = cv2.VideoCapture(i)
            ret, frame = cap.read()
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cap.release()
            cv2.destroyAllWindows()
            n += 1
        except:
            cap.release()
            cv2.destroyAllWindows()
            break
    return n

def sendP(message):
    ser = serial.Serial('COM7', 9600, timeout=0.5)
    possible_val = [0,1,2,3]
    if int(message) in possible_val:
        val = bytes(message,"utf-8")
        while True:
            ser.write(val)
            if str(ser.read(10).decode()) == '5':
               break
