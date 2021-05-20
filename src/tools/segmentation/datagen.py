
import cv2
import json
import tqdm
import numpy as np
from fire import Fire
from pathlib import Path

# raw image, anno -> rescaled_image, rescaled_anno -> img, heatmap


from src.core.constants import KEYPOINTS
from src.core.utils import rescale_anno, rescale_img


TARGET_RESOLUTION = (1280, 720)

KEYPOINT_RADIUS = 20


def id2color(keypoint_id: int):
    return keypoint_id + 1


def color2id(color: int):
    return color - 1


class DatasetGenerator:

    def __init__(self, dataset_root: str):

        self.dataset_root = Path(dataset_root)

        self.raw_data_dir = self.dataset_root / 'raw'
        self.target_dir = self.dataset_root / 'segmentation'

        self.target_img_dir = self.target_dir / 'img'
        self.target_mask_dir = self.target_dir / 'mask'

        self.target_img_dir.mkdir(exist_ok=True, parents=True)
        self.target_mask_dir.mkdir(exist_ok=True, parents=True)

    def get_mask(self, anno, resolution):
        w, h = resolution
        mask = np.zeros((h, w))
        for name, pos in anno.items():
            x, y = pos
            color = id2color(KEYPOINTS[name]['id'])
            mask = cv2.circle(mask, center=(x, y), radius=KEYPOINT_RADIUS, color=color, thickness=-1)
        return mask

    def generate(self):

        img_list = [x for x in self.raw_data_dir.glob('*/*.png')]

        for img_path in tqdm.tqdm(img_list):

            anno_path = img_path.parent / 'annotation' / f'{img_path.stem}.json'

            if anno_path.exists():
                img = cv2.imread(str(img_path))

                h, w, _ = img.shape

                with open(str(anno_path), 'r') as file:
                    anno = json.load(file)

                anno = rescale_anno(anno, src_resolution=(w, h), des_resolution=TARGET_RESOLUTION)
                img = rescale_img(img, resolution=TARGET_RESOLUTION)

                mask = self.get_mask(anno, resolution=TARGET_RESOLUTION)

                target_img_path = self.target_img_dir / img_path.name
                target_mask_path = self.target_mask_dir / img_path.name

                cv2.imwrite(str(target_img_path), img)
                cv2.imwrite(str(target_mask_path), mask)


def main(dataset_root: str):
    """
    Generate dataset for image segmentation problem
    :param dataset_root:
    :return:
    """
    generator = DatasetGenerator(dataset_root)
    generator.generate()


if __name__ == '__main__':
    Fire(main)
