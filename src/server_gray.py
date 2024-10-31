from flask import Flask, render_template, Response
import time
import cv2
import numpy as np
import os
from ultralytics import YOLO

#iou บอกตำแหน่งที่ detect ว่าเจอมาถูกไหม
#อธิบาย structure ทำไมถึงเลือก ทำไมข้อมูลเท่านี้ ต้องเทียบกับ schitech แบบนี้
app = Flask(__name__)

average_velocity_y_mm_s = 0.0

crack_detected = False

@app.route('/')
def index():
    return render_template('index.html')

def calculate_velocity(frame, prev_y_positions, frame_count):

    global average_velocity_y_mm_s
    X, Y, width, height = 203, 0, 450, 360
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    groi = gray[Y:Y+height, X:X+width]
    roi = frame[Y:Y+height, X:X+width]
    _, threshold = cv2.threshold(groi, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pixels_to_mm = 125 / 341
    frame_rate = 15 

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 0:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 0), 2)

            center_y = y + h // 2
           
            current_y_position = center_y
            prev_y_positions.append(current_y_position)

    if len(prev_y_positions) > 1:
        frame_count += 1

        if frame_count > 1:
            total_displacement_y = sum(abs(prev_y_positions[i] - prev_y_positions[i - 1]) for i in range(1, len(prev_y_positions)))
            average_velocity_y_mm_s = (total_displacement_y * pixels_to_mm) / ((frame_count - 1) / frame_rate)
    return prev_y_positions, frame_count

def generate_frames():

    global crack_detected, average_velocity_y_mm_s
    
    VIDEOS_DIR = os.path.join('.', 'video')
    video_path = os.path.join(VIDEOS_DIR, '141209.5.avi')

    prev_y_positions = [] 
    frame_count = 0  

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

        prev_y_positions, frame_count = calculate_velocity(frame, prev_y_positions, frame_count)

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

@app.route('/velocity_feed')
def velocity_feed():
    def generate_velocity():
        global average_velocity_y_mm_s
        while True:
            yield f"data:{average_velocity_y_mm_s:.2f} mm/s\n\n"
            # time.sleep(1) 
    return Response(generate_velocity(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
