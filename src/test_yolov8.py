import cv2
from ultralytics import YOLO

model = YOLO("data/Pistols.v1-resize-416x416.yolov8/yolov8n.pt")

cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()

    results = model(frame,show=True)
    print(results)