from scipy.spatial import distance as dist
import numpy as np

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

def centroid(x,y):
    return (sum(x) / len(x), sum(y) / len(y))

def find_center(coords):
    return np.array(centroid(*coords.squeeze().T), dtype="int32")

def scale_corners(corners, center_coords, multiple=1.3):
    return (((corners - center_coords)  * multiple) + center_coords).astype("int32")

def get_vertical_stretch_amount(corners, divisor=5):
    squeezed_corners = corners.squeeze()
    sorted_by_y = squeezed_corners[np.argsort(squeezed_corners[:, 1]), :]
    return (sorted_by_y[-1][1] - sorted_by_y[0][1]) // divisor

# this function assumes that corners has already been sorted like so:
# [[tl], [bl], [br], [tr]]
def stretch_vertically(sorted_corners, vertical_stretch_amount):
    (tl, bl, br, tr) = sorted_corners.squeeze()

    tl = [tl[0], tl[1] + vertical_stretch_amount]
    tr = [tr[0], tr[1] + vertical_stretch_amount]
    bl = [bl[0], bl[1] - vertical_stretch_amount]
    br = [br[0], br[1] - vertical_stretch_amount]
    return np.array([[tl], [bl], [br], [tr]], dtype="int32")

def stretch_and_scale(corners):
    center_coords = find_center(corners)
    scaled_corners = scale_corners(corners, center_coords)
    vertical_stretch_amount = get_vertical_stretch_amount(scaled_corners)
    sorted_corners = corner_sorter(scaled_corners)
    vertically_stretched_corners = stretch_vertically(sorted_corners, vertical_stretch_amount)
    return vertically_stretched_corners