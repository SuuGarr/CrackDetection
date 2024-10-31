import os
from ultralytics import YOLO
import glob
import cv2

image_path = "F:\\2.detection-2ndData\\newdata_test\\thermaltestAll2"
output_path = "F:\\2.detection-2ndData\\predic_test"

if not os.path.exists(output_path):
    os.makedirs(output_path)
    print("Created output directory", output_path)

model_path = os.path.join('.', 'runs', 'train', 'weights', 'best.pt')
model = YOLO(model_path)

true_positives, false_positives, false_negatives = 0, 0, 0

for image_file in glob.glob(os.path.join(image_path, '*.bmp')):
    image = cv2.imread(image_file)
    if image is None:
        print(f"Failed to read the image : {image_file}")
        continue

    print(f"Processing image: {image_file}")

    prediction = model(image)[0]

    print(f"Predictions: {prediction}")

    if prediction is None or len(prediction) == 0:
        print(f"No detections for {image_file}")
    else:
        print(f"Detections found for {image_file}: {prediction}")
    
    print(f"Number of detection : {len(prediction)}")
    print(f"Detections for {image_file} : {prediction}")

    base_name = os.path.splitext(os.path.basename(image_file))[0]
    txt_path = os.path.join(output_path, f"{base_name}.txt")

    with open(txt_path, 'w') as txt_file: 
        detection_found = False
        threshold = 0.5 

        if hasattr(prediction, 'boxes') and prediction.boxes:
            for box in prediction.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = box
                if score > threshold:
                    detection_found = True
                    detection_str = f"{class_id} {x1} {y1} {x2} {y2} {score}\n"
                    txt_file.write(detection_str)

        if not detection_found:
            print(f"No detections found or 'boxes' attribute missing for image: {image_file}") 

print("Processing Complete.", output_path)
