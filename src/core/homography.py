
import cv2
import json
import numpy as np
from pathlib import Path

from src.core.constants import CONNECTIONS
from src.core.utils import initialize_field_template, TEMPLATE_RESOLUTION


class Homography:

    def __init__(self, image_keypoints, resolution):

        self.width, self.height = resolution

        self.field_points = []
        self.image_points = []

        self.field_template = initialize_field_template(TEMPLATE_RESOLUTION)

        for name, pos in image_keypoints.items():
            img_x, img_y = pos
            self.image_points.append([img_x, img_y])
            field_x, field_y = self.field_template[name]
            self.field_points.append([field_x, field_y])

        self.H, status = cv2.findHomography(np.array(self.image_points), np.array(self.field_points))

        print(self.H)

    def warp(self, image):

        h, w, c = image.shape

        s = cv2.warpPerspective(image, self.H, dsize=TEMPLATE_RESOLUTION, borderValue=1)

        for connection in CONNECTIONS:
            a, b = connection

            pt1 = self.field_template[a]
            pt2 = self.field_template[b]

            pt1 = (int(pt1[0]), int(pt1[1]))
            pt2 = (int(pt2[0]), int(pt2[1]))

            cv2.line(s, pt1=pt1, pt2=pt2, color=(0, 255, 0), thickness=2)

        for connection in CONNECTIONS:
            a, b = connection

            x1, y1 = self.field_template[a]
            x2, y2 = self.field_template[b]

            Hinv = np.linalg.inv(self.H)

            pt1 = cv2.perspectiveTransform(np.array([[[x1, y1]]]), Hinv)
            pt2 = cv2.perspectiveTransform(np.array([[[x2, y2]]]), Hinv)

            pt1 = pt1[0][0]
            pt2 = pt2[0][0]

            pt1 = (int(pt1[0]), int(pt1[1]))
            pt2 = (int(pt2[0]), int(pt2[1]))

            cv2.line(image, pt1, pt2, color=(0, 255, 0), thickness=2)

        cv2.imwrite('test.png', s)
        cv2.imwrite('test2.png', image)


if __name__ == '__main__':

    path = Path('/home/andrii/Desktop/diploma_data/DONE/GWzkF1hNavc_1')

    entry_name = 'GWzkF1hNavc_0009296'

    img_path = path / f'{entry_name}.png'
    anno_path = path / 'annotation' / f'{entry_name}.json'

    with open(str(anno_path), 'r') as file:
        keypoints = json.load(file)

    image = cv2.imread(str(img_path))
    height, width, _ = image.shape

    homography = Homography(keypoints, resolution=(width, height))


    homography.warp(image)





