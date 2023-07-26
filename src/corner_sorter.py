from scipy.spatial import distance as dist
import numpy as np
import cv2
def corner_sorter(pts):
    pts = pts.squeeze()
    xSorted = pts[np.argsort(pts[:, 0]), :]

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (bl, tl) = leftMost
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (br, tr) = rightMost
    
    return np.array([[tl], [bl], [br], [tr]], dtype="int32")