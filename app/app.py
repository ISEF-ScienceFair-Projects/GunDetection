from flask import Flask, render_template, Response
import cv2


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    while True:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + camera.get_frame() + b'\r\n')


class Camera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return b''
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        return frame
    
camera = Camera()




@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
