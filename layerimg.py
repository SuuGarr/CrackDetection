import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import os
import torch

model_path = os.path.join('.', 'runs', 'train4', 'weights', 'best.pt')
model = YOLO(model_path)

first_conv_layer = None
for name, module in model.model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        first_conv_layer = module
        break
if first_conv_layer is not None:

    filters = first_conv_layer.weight.data.cpu().numpy()
    filter_count = filters.shape[0]

    fig, axarr = plt.subplots(1, filter_count)
    for idx in range(filter_count):
        axarr[idx].imshow(filters[idx][0], cmap='gray')  
        axarr[idx].axis('off')

    save_path = 'D:\\1.detection\\layers\\conv_filters.png'
    plt.savefig(save_path, bbox_inches='tight') 
    plt.show() 
else:
    print("No convolutional layer found")