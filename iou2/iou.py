import cv2
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv

prediction_path = "F:\\2.detection-2ndData\\predic_test"
test_labels_path = "F:\\2.detection-2ndData\\newdata_test\\annotated"
test_images_path = "F:\\2.detection-2ndData\\newdata_test\\thermaltestAll2"  
result_path = "F:\\2.detection-2ndData\\result_test\\iou"
metrics_path = "F:\\2.detection-2ndData\\result_test"

if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(metrics_path):
    os.makedirs(metrics_path)

img_width, img_height = 640, 480 

ious = []

PRED_LABEL = 'Model Pred'
TRUTH_LABEL = 'Ground Truth'

iou_threshold = 0.5
total_true_positives, total_false_positives, total_false_negatives, total_true_negatives = 0, 0, 0, 0

def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    combined_area = box1_area + box2_area - intersection_area
    iou = intersection_area / combined_area if combined_area != 0 else 0
    return iou

def convert_to_absolute(norm_box, img_width, img_height):

    center_x, center_y, width, height = norm_box
    x1 = (center_x - width / 2) * img_width
    y1 = (center_y - height / 2) * img_height
    x2 = (center_x + width / 2) * img_width
    y2 = (center_y + height / 2) * img_height
    return [x1, y1, x2, y2]

def draw_boxes(image, boxes, color, label, ious=None):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        if ious is not None and i < len(ious):
            iou_text = f"IOU: {ious[i]:.2f}"
            text_size = cv2.getTextSize(iou_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = x1 - text_size[0] - 10  
            rect_width = text_size[0] + 10
            cv2.rectangle(image, (text_x, y1 - 20), (text_x + rect_width, y1 - 5), (255, 255, 255), -1)
            cv2.putText(image, iou_text, (text_x, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (3, 252, 65), 2)

        label_offset = 15 if label == PRED_LABEL else -25  
        label_x = x1 + 30 if label == PRED_LABEL else x1  
        cv2.putText(image, label, (label_x, max(y1 + label_offset, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

for img_file in glob.glob(os.path.join(test_images_path, '*.bmp')):
    base_name = os.path.splitext(os.path.basename(img_file))[0]
    pred_file = os.path.join(prediction_path, base_name + '.txt')
    truth_file = os.path.join(test_labels_path, base_name + '.txt')

    image = cv2.imread(img_file) if os.path.exists(img_file) else None
    if image is None:
        continue  # Skip if the image does not exist

    pred_boxes = []
    if os.path.exists(pred_file):
        with open(pred_file, 'r') as f:
            pred_boxes = [list(map(float, line.split()[1:5])) for line in f.readlines()]

    truth_boxes = []
    if os.path.exists(truth_file):
        with open(truth_file, 'r') as f:
            truth_boxes = [convert_to_absolute(list(map(float, line.split()[1:5])), img_width, img_height) for line in f.readlines()]

    # Draw predicted and truth boxes on the images
    individual_ious = []
    for pred_box in pred_boxes:
        max_iou = 0
        for truth_box in truth_boxes:
            iou = calculate_iou(pred_box, truth_box)
            max_iou = max(max_iou, iou)
        ious.append(max_iou)
        individual_ious.append(max_iou)

    draw_boxes(image, pred_boxes, (37, 245, 252), PRED_LABEL, individual_ious)
    draw_boxes(image, truth_boxes, (236, 92, 255), TRUTH_LABEL)
    cv2.imwrite(os.path.join(result_path, base_name + '_iou.bmp'), image)

    # Evaluate the detection results
    for truth_box in truth_boxes:
        match_found = any(calculate_iou(pred_box, truth_box) >= iou_threshold for pred_box in pred_boxes)
        if match_found:
            total_true_positives += 1
        else:
            total_false_negatives += 1

    total_false_positives += len(pred_boxes) - sum(1 for iou in individual_ious if iou >= iou_threshold)
    # Merge the if statement with the enclosing one
    if not truth_boxes and not pred_boxes:
        total_true_negatives += 1  # Increment for correct no detections in 'no_crack' images

# Calculate and print metrics
average_iou = np.mean(ious) if ious else 0
total_predictions = total_true_positives + total_false_positives
total_actuals = total_true_positives + total_false_negatives
precision = total_true_positives / total_predictions if total_predictions > 0 else 0
recall = total_true_positives / total_actuals if total_actuals > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

# Normalized confusion matrix calculations
norm_confusion_matrix = np.array([
    [total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0,
     total_false_positives / (total_false_positives + total_true_negatives) if (total_false_positives + total_true_negatives) > 0 else 0],
    [total_false_negatives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0,
     total_true_negatives / (total_false_positives + total_true_negatives) if (total_false_positives + total_true_negatives) > 0 else 0]
])

csv_file_path = os.path.join(metrics_path, 'performance_metrics.csv')
with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Metric', 'Value'])
    writer.writerow(['Precision', '{:.4f}'.format(precision)])
    writer.writerow(['Recall', '{:.4f}'.format(recall)])
    writer.writerow(['F1 Score', '{:.4f}'.format(f1_score)])
    writer.writerow(['Average IoU', '{:.4f}'.format(average_iou)])

# Print metrics to console
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print(f"Average IoU: {average_iou:.4f}")

# Construct standard confusion matrix for visualization
confusion_matrix = np.array([
    [total_true_positives, total_false_positives],
    [total_false_negatives, total_true_negatives]
])

# Plot and save standard Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap='Blues',
            xticklabels=['Actual Positive', 'Actual Negative'],
            yticklabels=['Predicted Positive', 'Predicted Negative'])
plt.title('Confusion Matrix')
plt.ylabel('Predicted label')
plt.xlabel('Actual label')
confusion_matrix_file = os.path.join(metrics_path, 'confusion_matrix.png')
plt.savefig(confusion_matrix_file)
plt.close()

# Plot and save Normalized Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(norm_confusion_matrix, annot=True, fmt=".2f", cmap='Blues',
            xticklabels=['Actual Positive', 'Actual Negative'],
            yticklabels=['Predicted Positive', 'Predicted Negative'])
plt.title('Normalized Confusion Matrix')
plt.ylabel('Predicted label')
plt.xlabel('Actual label')
norm_confusion_matrix_file = os.path.join(metrics_path, 'normalized_confusion_matrix.png')
plt.savefig(norm_confusion_matrix_file)
plt.close()

print("Metrics and confusion matrices saved in", metrics_path)