from typing import List, Dict
import serial
import cv2

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
