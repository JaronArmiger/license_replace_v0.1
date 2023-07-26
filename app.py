import sys
import torch
import numpy as np
import os
import matplotlib.pyplot as plt 
import cv2
from segment_anything import sam_model_registry, SamPredictor
import math

if len(sys.argv) != 2:
    print('provide one command line argument (path to image)')
    sys.exit()

image = cv2.imread('./images/car17.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

best_weights = './static/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', best_weights, trust_repo=True)


results = model(image)

results_df = results.pandas().xyxy[0].loc[0]


x_min = int(results_df['xmin'])
x_max = int(results_df['xmax'])
y_min = int(results_df['ymin'])
y_max = int(results_df['ymax'])

print(results_df)