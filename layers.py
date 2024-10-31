from ultralytics import YOLO
import os

model_path = os.path.join('.', 'runs', 'train4', 'weights', 'best.pt')
model = YOLO(model_path)

print(model.model)

layer_count = 0

for layer in model.model.modules():
    layer_count += 1
    
print(f"The model contains {layer_count} layers.")

