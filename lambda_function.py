import sys
import torch
import numpy as np
import matplotlib.pyplot as plt 
import cv2
from segment_anything import sam_model_registry, SamPredictor
from src.utils import read_s3_image, write_s3_image, stretch_and_scale
from yolov5.models.yolo import Model

bucket_name = "license-replace"

def handler(event, context):
    bucket_name = event.bucket_name

    logo_image_path = event.logo_image_path or './static/allo_logo_rounded_border_01.png'
    car_image_path = event.car_image_path

    image = read_s3_image(bucket_name, car_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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
    try:
        while (min_distance >= 0):
            if (len(corners) == 4):
                break
            min_distance -= 5
            feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=min_distance, blockSize=9)
            corners = cv2.goodFeaturesToTrack(mask0_inv, **feature_params)
            counter = counter + 1
    except:
        print('could not detect corners of license plate')
        sys.exit()

    if len(corners)!= 4:
        print('could not detect corners of license plate')
        sys.exit()

    modified_corners = stretch_and_scale(corners)


    allo_logo = cv2.imread(logo_image_path)
    rounded_mask_white = cv2.imread('./static/rounded_mask_white.png')

    bottom_left = [0, allo_logo.shape[0]]
    top_left = [0,0]
    bottom_right = [allo_logo.shape[1], allo_logo.shape[0]]
    top_right = [allo_logo.shape[1], 0]
    pts_src = np.array([bottom_left, top_left, top_right, bottom_right])

    h, status = cv2.findHomography(pts_src, modified_corners, cv2.RANSAC)
    plate_isolated = cv2.warpPerspective(allo_logo, h, (image.shape[1], image.shape[0]))

    rounded_mask_warped = cv2.warpPerspective(rounded_mask_white, h, (image.shape[1], image.shape[0]))
    mask_inv = cv2.cvtColor(cv2.bitwise_not(rounded_mask_warped), cv2.COLOR_BGR2GRAY)

    img_bo = cv2.bitwise_and(image, image, mask=mask_inv)
    img_bo = cv2.cvtColor(img_bo, cv2.COLOR_BGR2RGB)
    final = cv2.bitwise_or(img_bo, plate_isolated)

    result_image_string = cv2.imencode(".jpg", final)[1].tobytes()
    write_s3_image(result_image_string, bucket_name, "result.jpg")
