from ultralytics import YOLO
import torch
import os

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    print(" CUDA is not available")

model = YOLO("yolov8n.yaml")

if __name__ == '__main__':
    results = model.train(data="config.yaml", epochs=100, device= device, project='F:\\2.detection-2ndData\\runs') 