from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time
import io
import base64
from PIL import Image
import cv2
import numpy as np
import pyshine as ps
from flask_cors import CORS, cross_origin
from engineio.payload import Payload

Payload.max_decode_packets = 2048
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string = base64_string[idx + 7:]

    sbuf = io.BytesIO()
    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)
    
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

def moving_average(x):
    return np.mean(x)

@socketio.on('catch-frame')
def catch_frame(data):
    emit('response_back', data)

global fps, prev_recv_time, cnt, fps_array
fps = 60
prev_recv_time = 0
cnt = 0
fps_array = [0]

@socketio.on('image')
def image(data_image):
    global fps, cnt, prev_recv_time, fps_array
    recv_time = time.time()
    text = 'FPS: ' + str(fps)
    frame = readb64(data_image)
    
    # insert logic here
    """
    frame = gundetection...
    """
    
    frame = ps.putBText(frame, text, text_offset_x=20, text_offset_y=30, vspace=20, hspace=10, font_scale=1.0, background_RGB=(10, 20, 222), text_RGB=(255, 255, 255))
    imgencode = cv2.imencode('.jpeg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])[1]

    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData

    emit('response_back', stringData)

    fps = 1 / (recv_time - prev_recv_time)
    fps_array.append(fps)
    fps = round(moving_average(np.array(fps_array)), 1)
    prev_recv_time = recv_time
    
    cnt += 1
    if cnt == 30:
        fps_array = [fps]
        cnt = 0

if __name__ == '__main__':
    socketio.run(app, port=9990, debug=True)
