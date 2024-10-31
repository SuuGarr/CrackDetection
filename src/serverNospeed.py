from flask import Flask, render_template, Response
import time
import cv2
import numpy as np
import os
from ultralytics import YOLO

app = Flask(__name__)

crack_detected = False

@app.route('/')
def index():
    return render_template('index.html')
def generate_frames():

    global crack_detected
    
    VIDEOS_DIR = os.path.join('.', 'video')
    video_path = os.path.join(VIDEOS_DIR, '02_50A_4.7.avi')

    cap = cv2.VideoCapture(video_path)

    if not os.path.exists(video_path):
        print(f"Video file does not exist: {video_path}")
        return
        
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    model_path = os.path.join('.', 'runs', 'train', 'weights', 'best.pt')
    model = YOLO(model_path)
    threshold = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 80, 180), 2)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 180), 2)
                crack_detected = True

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status_feed')
def status_feed():
    def generate_status():
        global crack_detected
        while True:
            status = "NG" if crack_detected else "OK"
            yield f"data:{status}\n\n"
            time.sleep(1)

    return Response(generate_status(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
