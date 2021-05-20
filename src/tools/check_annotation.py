
import cv2
import json
from pathlib import Path


def check(data_dir, target_dir):

    anno_dir = Path(data_dir) / 'annotation'

    for img_path in Path(data_dir).glob('./*.png'):
        img_name = img_path.stem
        anno_file = anno_dir / f'{img_name}.json'

        with open(str(anno_file), 'r') as file:
            anno = json.load(file)

        image = cv2.imread(str(img_path))

        for name, pos in anno.items():
            x, y = pos
            cv2.circle(image, center=(x,y), radius=5, color=(0, 255, 0), thickness=5)

        target_path = Path(target_dir) / f'{img_name}.png'

        cv2.imwrite(str(target_path), image)

check('/home/andrii/Desktop/diploma_data/GRpRGZHJFhU', '/home/andrii/Desktop/DEBUG')