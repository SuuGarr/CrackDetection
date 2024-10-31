import cv2
import numpy as np
import os
from ultralytics import YOLO

VIDEOS_DIR = os.path.join('.', 'video')
video_path = os.path.join(VIDEOS_DIR, '08.avi')
video_path_out = f'{video_path}_out.avi'

model_path = os.path.join('.', 'runs', 'train', 'weights', 'best.pt')
model = YOLO(model_path)
threshold = 0.5

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

if not ret:
    print("Failed to read the video")
    exit()

H, W, _ = frame.shape  
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'XVID'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

velocity_file_path = os.path.join(os.getcwd(), 'velocity.txt')

X, Y, width, height = 203, 50, 450, 300  
prev_y_positions = []
total_displacement_y = 0
frame_count = 0
frame_rate = 15

pixels_to_mm = 125 / 341

while ret:
    results = model(frame)[0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    groi = gray[Y:Y+height, X:X+width]
    roi = frame[Y:Y+height, X:X+width]
    
    retval, Threshold = cv2.threshold(groi, 210 , 255, cv2.THRESH_BINARY)
    contour, _ = cv2.findContours(Threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contour:
        largest_contour = max(contour, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        max_roi = frame[Y+y:Y+y+h, X+x:X+x+w]
    
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 80, 180), 2)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 180), 2, cv2.LINE_AA)
    for cnt in contour:
        area = cv2.contourArea(cnt)
        if area > 0:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 0), 2)
            
            center_x = x + w // 2
            center_y = y + h // 2
           
            current_y_position = center_y
            prev_y_positions.append(current_y_position)

    if len(prev_y_positions) > 1:
        absolute_displacement_y = abs(prev_y_positions[-1] - prev_y_positions[-2])
        if frame_count > 1:
            time_seconds = frame_count / frame_rate
            velocity_y_mm_s = absolute_displacement_y / time_seconds
        else:
            velocity_y_mm_s = 0
        
        total_displacement_y += absolute_displacement_y
        frame_count += 1

        if frame_count > 1:
            average_velocity_y_mm_s = (total_displacement_y * pixels_to_mm) / (frame_count / frame_rate)
            print(f"Velocity: {average_velocity_y_mm_s:.2f} mm/s")
            cv2.putText(roi, f"Average Velocity: {average_velocity_y_mm_s:.2f} mm/s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        try:
            with open(velocity_file_path, 'w') as f:
                f.write(f"{average_velocity_y_mm_s}")  
            print(f"Successfully wrote velocity to {velocity_file_path}")
        except Exception as e:
            print(f"Failed to write velocity: {e}")    
    
    cv2.imshow('Frame', frame)
    out.write(frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()