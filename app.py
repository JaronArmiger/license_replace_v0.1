import sys
import torch
import numpy as np
import matplotlib.pyplot as plt 
import cv2
from segment_anything import sam_model_registry, SamPredictor
from src.corner_sorter import corner_sorter

if len(sys.argv) != 2:
    print('provide one command line argument (path to car image)')
    sys.exit()

image = cv2.imread(sys.argv[1])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


best_weights = './static/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', best_weights, trust_repo=True)


results = model(image)

results_df = results.pandas().xyxy[0].loc[0]


x_min = int(results_df['xmin'])
x_max = int(results_df['xmax'])
y_min = int(results_df['ymin'])
y_max = int(results_df['ymax'])


sam_checkpoint = "./static/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

predictor.set_image(image)

input_box = np.array([x_min, y_min, x_max, y_max])
masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
)

mask0 = masks[0]
mask0_inv = np.logical_not(mask0)
mask0_inv = mask0_inv*1
mask0_inv = mask0_inv.astype(np.uint8)

min_distance = 20

feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=min_distance, blockSize=9)
corners = cv2.goodFeaturesToTrack(mask0_inv, **feature_params)

counter = 1
while (min_distance >= 0):
  if (len(corners) == 4):
     break
  min_distance -= 5
  feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=min_distance, blockSize=9)
  corners = cv2.goodFeaturesToTrack(mask0_inv, **feature_params)
  counter = counter + 1


if len(corners)!= 4:
   print('could not detect corners of license plate')
   sys.exit()

sorted_corners = corner_sorter(corners)

allo_logo = cv2.imread('./static/allo_logo_flipped.png')

bottom_left = [0, allo_logo.shape[0]]
top_left = [0,0]
bottom_right = [allo_logo.shape[1], allo_logo.shape[0]]
top_right = [allo_logo.shape[1], 0]
pts_src = np.array([bottom_left, top_left, top_right, bottom_right])

h, status = cv2.findHomography(pts_src, sorted_corners)
plate_isolated = cv2.warpPerspective(allo_logo, h, (image.shape[1], image.shape[0]))

new_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
new_image = cv2.drawContours(new_mask, [sorted_corners], 0, 255, -1)
new_mask_inv = cv2.bitwise_not(new_mask)

img_bo = cv2.bitwise_and(image, image, mask=new_mask_inv)
img_bo = cv2.cvtColor(img_bo, cv2.COLOR_BGR2RGB)
final = cv2.bitwise_or(img_bo, plate_isolated)

cv2.imwrite('./static/results/original.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.imwrite('./static/results/result_hide.jpg', img_bo)
cv2.imwrite('./static/results/result_replace.jpg', final)