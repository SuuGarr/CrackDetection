from flask import Flask, render_template, Response, request, jsonify
import time
import cv2
import numpy as np
import os
from ultralytics import YOLO
from predictPara import predict_parameters

app = Flask(__name__)

average_velocity_y_mm_s = 0.0
predicted_current_value = 0.0
predicted_velocity_y_mm_s = 0.0
input_current_value = 0

crack_detected = False
current_set = False

@app.route('/')
def index():
    return render_template('index.html')

def calculate_velocity(frame, prev_y_positions, frame_count):
    global average_velocity_y_mm_s
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Defining HSV ranges for high temperatures (white and red)
    lower_white = np.array([0, 0, 212], dtype=np.uint8)
    upper_white = np.array([180, 30, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Red has two ranges because it wraps around the hue wheel
    lower_red1 = np.array([0, 70, 50], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 70, 50], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Combine masks for white and red
    combined_mask = cv2.bitwise_or(white_mask, red_mask)

    # Find contours in the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        center_y = y + h // 2

        # Visualize for debugging
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

        prev_y_positions.append(center_y)

    if len(prev_y_positions) > 1:
        frame_count += 1
        if frame_count > 1:
            total_displacement_y = sum(abs(prev_y_positions[i] - prev_y_positions[i - 1]) for i in range(1, len(prev_y_positions)))
            average_velocity_y_mm_s = (total_displacement_y * 125 / 322) / ((frame_count - 1) / 15)

    return prev_y_positions, frame_count

def update_predicted_velocity():
    global predicted_velocity_y_mm_s, input_current_value, average_velocity_y_mm_s

    if input_current_value != 0:
        _, predicted_velocity_y_mm_s = predict_parameters(input_current_value, average_velocity_y_mm_s)


def generate_frames():
    global crack_detected, average_velocity_y_mm_s, predicted_current_value, predicted_velocity_y_mm_s, current_set

    if not current_set:
        print("Current value not set, stopping video feed.")
        yield b'Current value not set or invalid.\r\n'
        return

    VIDEOS_DIR = os.path.join('.', 'video')
    video_path = os.path.join(VIDEOS_DIR, '02_45A_5.3.avi')

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
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                crack_detected = True

        predicted_current_value, predicted_velocity_y_mm_s = predict_parameters(input_current_value, average_velocity_y_mm_s)
        update_predicted_velocity()

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
        global average_velocity_y_mm_s, predicted_current_value, predicted_velocity_y_mm_s
        while True:
            data = (f"Average Speed: {average_velocity_y_mm_s:.2f} mm/s, "
                    f"Predicted Speed: {predicted_velocity_y_mm_s:.2f} mm/s, "
                    f"Predicted Current: {predicted_current_value:.2f}A")
            yield f"data:{data}\n\n"
            time.sleep(1)
    return Response(generate_velocity(), mimetype='text/event-stream')

@app.route('/submit_current', methods=['POST'])
def submit_current():
    global input_current_value, current_set, crack_detected, predicted_current_value, predicted_velocity_y_mm_s
    current_value = request.form.get('currentValue')
    print(f"Input Value :{current_value}")
    if not current_value or int(current_value) == 0:
        return jsonify({"error": "Please input welding data"}), 400

    current_set = True
    input_current_value = int(current_value)

    # Predict parameters using the static input current value
    if crack_detected:
        predicted_current_value, predicted_velocity_y_mm_s = predict_parameters(input_current_value, average_velocity_y_mm_s)
        print(f"Predicted Current (Crack Detected): {predicted_current_value} A, Predicted Velocity: {predicted_velocity_y_mm_s} mm/s")
        return jsonify({"current": predicted_current_value, "status": "crack"})
    
    else:
        # Return input current value and static average speed as predicted values if no crack is detected
        predicted_current_value = input_current_value
        predicted_velocity_y_mm_s = average_velocity_y_mm_s
        print(f"Input Current: {input_current_value} A, Predicted Velocity: {predicted_velocity_y_mm_s} mm/s")
        return jsonify({"current": input_current_value, "status": "no_crack"})



if __name__ == '__main__':
    app.run(debug=True, port=5000)
