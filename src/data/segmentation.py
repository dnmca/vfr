
import cv2
import json
import keras
import numpy as np
from tqdm import tqdm
from pathlib import Path

from src.core.constants import KEYPOINTS
from src.core.utils import rescale_img, rescale_keypoints


IMAGE_SIZE = (512, 512)
CHANNELS = 3
CLASSES = 10


class SegmentationDatasetGenerator(keras.utils.Sequence):

    def __init__(self, indices, idx2filename, idx2keypoints, batch_size, shuffle):

        self.indices = indices
        self.idx2filename = idx2filename
        self.idx2keypoints = idx2keypoints

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle is True:
            np.random.shuffle(self.indices)

    def __len__(self):
        """Number of batches per epoch"""
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        """Get batch by index"""
        indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        batch_x, batch_y = self._get_data(indices)
        return batch_x, batch_y

    def _get_data(self, indices):
        x, y = [], []
        for index in indices:
            img = self._load_image(self.idx2filename[index])
            keypoints = self.idx2keypoints[index]
            heatmap = self._get_heatmap(img, keypoints, sigma=13)
            x.append(img)
            y.append(heatmap)

        return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

    def _load_image(self, file_path):
        img = cv2.imread(file_path)
        img = rescale_img(img, resolution=IMAGE_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255
        return img

    def _get_heatmap(self, img, keypoints, sigma):
        img_h, img_w, _  = img.shape

        mask = np.zeros((img_h, img_w, CLASSES))
        xv = np.arange(img_w)
        yv = np.arange(img_h)
        xx, yy = np.meshgrid(xv, yv)

        for kp_name, kp_info in KEYPOINTS.items():
            kp_id = kp_info['id']
            if kp_name in keypoints:
                x, y = keypoints[kp_name]
                gaussian = (yy - y) ** 2
                gaussian += (xx - x) ** 2
                gaussian *= -1
                gaussian = gaussian / (2 * sigma ** 2)
                gaussian = np.exp(gaussian)
                mask[:, :, kp_id] = gaussian

        heatmap = np.reshape(mask, newshape=(img_h * img_w * CLASSES, 1))
        return heatmap


def get_test_data(dataset_path):
    raise NotImplementedError


def get_data(dataset_path, batch_size):

    dataset_path = Path(dataset_path)

    raw_data_path = dataset_path / 'raw'
    test_txt_path = dataset_path / 'test.txt'

    test_images = []

    with open(str(test_txt_path), 'r') as file:
        for line in file:
            test_images.append(line.strip())

    train_entries, test_entries = [], []

    for img_path in tqdm(raw_data_path.glob('*/*.png')):
        anno_path = img_path.parent / 'annotation' / f'{img_path.stem}.json'

        if not anno_path.exists():
            continue

        img = cv2.imread(str(img_path))
        img_h, img_w, _ = img.shape

        with open(str(anno_path), 'r') as file:
            keypoints = json.load(file)

        keypoints = rescale_keypoints(keypoints, src_resolution=(img_w, img_h), des_resolution=IMAGE_SIZE)

        if str(img_path.name) in test_images:
            test_entries.append((str(img_path), keypoints))
        else:
            train_entries.append((str(img_path), keypoints))

    indices = []
    idx2filepath = {}
    idx2keypoints = {}

    for idx in range(len(train_entries)):
        img_path, keypoints = train_entries[idx]
        indices.append(idx)
        idx2filepath[idx] = img_path
        idx2keypoints[idx] = keypoints

    np.random.shuffle(indices)

    split_point = int(0.9 * len(train_entries))

    train_indices = indices[:split_point]
    valid_indices = indices[split_point:]

    train_gen = SegmentationDatasetGenerator(
        train_indices,
        idx2filepath,
        idx2keypoints,
        batch_size=batch_size,
        shuffle=True
    )

    valid_gen = SegmentationDatasetGenerator(
        valid_indices,
        idx2filepath,
        idx2keypoints,
        batch_size=batch_size,
        shuffle=True
    )

    return train_gen, valid_gen
