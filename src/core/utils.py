
import cv2
import numpy as np

from src.core.constants import KEYPOINTS


TEMPLATE_RESOLUTION = (1280, 720)


def initialize_field_template(resolution) -> dict:

    width, height = resolution

    x_min = 0.25 * width
    x_max = 0.75 * width
    y_min = 0.25 * height
    y_max = 0.75 * height

    keypoint2pos = {}

    field_width = max([point['x'] for point in KEYPOINTS.values()])
    field_height = max([point['y'] for point in KEYPOINTS.values()])

    for name, info in KEYPOINTS.items():
        x_frac = info['x'] / field_width
        y_frac = info['y'] / field_height
        x_pos = x_min + x_frac * abs(x_max - x_min)
        y_pos = y_min + y_frac * abs(y_max - y_min)
        keypoint2pos[name] = [x_pos, y_pos]

    return keypoint2pos


def rescale_keypoints(annotation: dict, src_resolution: (int, int), des_resolution: (int, int)) -> dict:
    rescaled = {}
    src_w, src_h = src_resolution
    des_w, des_h = des_resolution
    for name, pos in annotation.items():
        x, y = pos
        x_scaled = int(np.round(x * des_w / src_w))
        y_scaled = int(np.round(y * des_h / src_h))
        rescaled[name] = [x_scaled, y_scaled]
    return rescaled


def rescale_img(img, resolution):
    return cv2.resize(img, dsize=resolution, interpolation=cv2.INTER_LINEAR)
