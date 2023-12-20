from ultralytics import YOLO
import cv2

model = YOLO('/Users/abhiramasonny/Developer/Python/yolov8/runs/detect/train7/weights/best.pt')

vid = cv2.VideoCapture('data/additional_data/video_4.mp4') 

while(True): 
      
    ret, frame = vid.read() 
    results = model.predict(frame, imgsz=640, conf=0.5)
    names = model.names
    for result in results: 

        boxes = result.boxes.cpu().numpy()
        for i,box in enumerate(boxes):                               # iterate boxes
            r = box.xyxy[0].astype(int)                              # get corner
            cv2.rectangle(frame, r[:2], r[2:], (255, 255, 255), 2)   # draw boxes on img
            cv2.putText(frame, "gun", r[2:]+2, 1,1.5,(0, 0, 200),2)
            
    cv2.imshow('frame', frame) 
      

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

vid.release()
cv2.destroyAllWindows()  