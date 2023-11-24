import cv2
import numpy as np
import base64

weight_path = 'yolo-obj_last.weights'
config_path = 'gun.cfg'

net = cv2.dnn.readNet(weight_path, config_path)

#gpu magic :)))
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
 
classes = ["Gun"]

layer_names = net.getLayerNames()
out_layers = net.getUnconnectedOutLayers()
if out_layers.ndim == 1:
    output_layers = [layer_names[i - 1] for i in out_layers]
else:
    output_layers = [layer_names[i[0] - 1] for i in out_layers]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

cap = cv2.VideoCapture(0)

tick = True
counter = 0
global finalconf

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            finalconf = confidence
            if confidence > 0.8:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 2)
    
    cv2.imshow("Camera Feed", img)
    if tick and len(boxes)>0:
        #save frame
        cv2.imwrite('use/gunImgs/frame.jpg', img)
        tick = False
        counter = 0
    counter+=1
    if counter == 100:
        counter = 0
        tick = True
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
