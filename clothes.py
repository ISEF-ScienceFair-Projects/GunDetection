import cv2
from cloth.cloth_detection import *
cap = cv2.VideoCapture(0)
model = Load_DeepFashion2_Yolov3()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break
    # + to tensor
    img_tensor = tf.convert_to_tensor(frame[tf.newaxis, ...], dtype=tf.float32) / 255.0
    img_with_boxes = Draw_Bounding_Box(frame, Detect_Clothes(img_tensor, model))
    cv2.imshow("Clothes detection", img_with_boxes)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()