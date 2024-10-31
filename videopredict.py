import cv2
import numpy as np
import os
from ultralytics import YOLO
import pandas as pd
import joblib  # Corrected import statement

# Define paths
VIDEOS_DIR = os.path.join('.', 'video')
video_path = os.path.join(VIDEOS_DIR, '08.avi')
video_path_out = f'{video_path}_out.avi'
model_path = os.path.join('.', 'runs', 'train7', 'weights', 'best.pt')

# Load the machine learning model
regressor = joblib.load('path_to_your_saved_model.pkl')  # Adjust the path to where your model is saved

# Setup YOLO model
model = YOLO(model_path)
threshold = 0.5

# Video setup
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Failed to read the video")
    exit()

H, W, _ = frame.shape  
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'XVID'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

velocity_file_path = os.path.join(os.getcwd(), 'velocity.txt')

X, Y, width, height = 203, 0, 450, 360  
prev_y_positions = []
total_displacement_y = 0
frame_count = 0
frame_rate = 15

pixels_to_mm = 125 / 341

while ret:
    results = model(frame)[0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    groi = gray[Y:Y+height, X:X+width]
    roi = frame[Y:Y+height, X:X+width]  # roi is defined here

    retval, Threshold = cv2.threshold(groi, 210 , 255, cv2.THRESH_BINARY)
    contour, _ = cv2.findContours(Threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contour:
        largest_contour = max(contour, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        max_roi = frame[Y+y:Y+y+h, X+x:X+x+w]
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw on roi within its scope

    # Ensure all references to roi are within this loop
    if len(prev_y_positions) > 1:
        absolute_displacement_y = abs(prev_y_positions[-1] - prev_y_positions[-2])
        frame_count += 1

        if frame_count > 1:
            average_velocity_y_mm_s = (total_displacement_y * pixels_to_mm) / (frame_count / frame_rate)
            predicted_speed = regressor.predict([[average_velocity_y_mm_s]])[0]
            
            # Use roi within its defined scope for displaying predicted speed
            cv2.putText(roi, f"Predicted Speed: {predicted_speed:.2f} mm/s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    out.write(frame)
    key = cv2.waitKey(30)
    if key == 27:
        break

    ret, frame = cap.read()

    key = cv2.waitKey(30)
    if key == 27:
        break

    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
