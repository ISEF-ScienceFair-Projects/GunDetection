from typing import List, Dict
import serial
import cv2
from src.yolov8.yoloProcessFrame import YoloObjD, run

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
import os
def extract_frames_from_video(video_path, output_directory, fps=60):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    cap = cv2.VideoCapture(video_path)

    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    skip_frames = original_fps // fps
    global captured_frames
    frame_number = 0
    captured_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_directory, f"frame_{captured_frames}.jpg")
        #print(frame_filename)
        cv2.imwrite(frame_filename, frame)
        captured_frames += 1

        frame_number += 1

    cap.release()

def runExtratFrames():
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, 'data/testing')
    frame_directory = os.path.join(current_directory, 'gunImages/frames')

    if not os.path.exists(frame_directory):
        os.makedirs(frame_directory)

    for file_name in os.listdir(data_directory):
        if file_name.endswith('.mp4'):
            video_path = os.path.join(data_directory, file_name)
            #output_directory = os.path.join(frame_directory,
    os.path.splitext(file_name)[0]
    extract_frames_from_video(video_path, frame_directory, 60)


def runOffofFrames():
    weight_path_gun = 'model/yolov8/5K_dataset_29epochs/runs/detect/train/weights/best.pt'
    
    for i in range(149):
        frame = cv2.imread(f"gunImages/frames/frame_{i}.jpg")
        yolo_detector = YoloObjD(weight_path_gun)
        yolo_detector.process_frame(frame)  


def fixPathToOneDirectory():
    for i in range(2,150):
        path = os.path.join("runs/detect 15-27-28-190", f"predict{i}/image0.jpg")
        cv2.imwrite("gunImages/frames_slower_more_accurate/frame_{}.jpg".format(i), cv2.imread(path))

#fixPathToOneDirectory()
import cv2
import os

def images_to_video(input_path, output_path, fps):
    images = [img for img in os.listdir(input_path) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(input_path, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(input_path, image)))

    cv2.destroyAllWindows()
    video.release()


