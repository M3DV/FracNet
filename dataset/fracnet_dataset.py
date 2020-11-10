import os
import random
from itertools import product
from numbers import Number
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from skimage.measure import regionprops
from torch.utils.data import DataLoader, Dataset


class FracNetDataset(Dataset):
    def __init__(self, info, crop=None, flip=False):
        self.info = info
        self.crop = crop
        self.flip = flip
        self.is_empty = False

    def __getitem__(self, idx):
        info  = self.info.iloc[idx]
        norm  = lambda x: 2 * (x+200) / 1200 - 1

        image = np.load(Path(info['image']))
        image = image[image.files[0]][None, ...]

        label = np.load(Path(info['label']))
        label = label[label.files[0]][None, ...]

        image = norm(np.clip(image, -200, 1000))

        #--------------------------transforms--------------------------#
        if self.crop is not None:
            index = [slice(None)] + self.rcorp(
                info['shape'], info['center'],
                self.crop['size'], self.crop['scale']
            )
            image, label = [x[tuple(index)] for x in [image, label]]

        if self.flip is not False:
            order = random.choices([1, -1], k=image.ndim-1)
            order = [slice(None, None, x) for x in [1] + order]
            image, label = [x[tuple(order)] for x in [image, label]]
        #--------------------------transforms--------------------------#

        image = image.astype(np.float32)
        label = np.where(label > 0, 1, 0).astype(np.float32)

        return image, label

    def __len__(self):
        return len(self.info)

    def rcorp(self, shape, center, size, scale=0):
        '''
        Args:
            shape: (D, H, W) or (H, W)
        Return:
            index: crop range for each dim
        '''

        if isinstance(size, Number):
            size = (size,) * len(center)

        if isinstance(scale, Number):
            scale = (scale,) * len(center)

        range = [x * y // 2 for x, y in zip(size, scale)]
        newcp = [x + random.randint(-y, y) for x, y in zip(center, range)]

        minp1 = [x - y // 2 for x, y in zip(newcp, size)]
        minp2 = [x if x > 0 else 0 for x in minp1]

        maxp1 = [x + y for x, y in zip(minp2, size)]
        maxp2 = [x if x < y else y for x, y in zip(maxp1, shape)]

        minp3 = [x - y for x, y in zip(maxp2, size)]
        index = [slice(x, y) for x, y in zip(minp3, maxp2)]

        return index


class FracNetNiiDataset(Dataset):

    def __init__(self, df, image_dir, label_dir=None, crop_size=64,
            transforms=None, num_samples=16, train=True):
        self.df = df
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.crop_size = crop_size
        self.transforms = transforms
        self.num_samples = num_samples
        self.train = train

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _get_pos_centroids(label_arr):
        centroids = [tuple([round(x) for x in prop.centroid])
            for prop in regionprops(label_arr)]

        return centroids

    @staticmethod
    def _get_symmetric_neg_centroids(pos_centroids, x_size):
        sym_neg_centroids = [(z, y, x_size - x) for z, y, x in pos_centroids]

        return sym_neg_centroids

    @staticmethod
    def _get_spine_neg_centroids(shape, crop_size, num_samples):
        z_min, z_max = crop_size // 2, shape[0] - crop_size // 2
        y_min, y_max = 300, 400
        x_min, x_max = shape[-1] // 2 - 40, shape[-1] // 2 + 40
        spine_neg_centroids = [(
            np.random.randint(z_min, z_max),
            np.random.randint(y_min, y_max),
            np.random.randint(x_min, x_max)
        ) for _ in range(num_samples)]

        return spine_neg_centroids

    def _get_neg_centroids(self, pos_centroids, image_shape):
        num_pos = len(pos_centroids)
        sym_neg_centroids = self._get_symmetric_neg_centroids(
            pos_centroids, image_shape[-1])

        if num_pos < self.num_samples // 2:
            spine_neg_centroids = self._get_spine_neg_centroids(image_shape,
                self.crop_size, self.num_samples - 2 * num_pos)
        else:
            spine_neg_centroids = self._get_spine_neg_centroids(image_shape,
                self.crop_size, num_pos)

        return sym_neg_centroids + spine_neg_centroids

    def _get_roi_centroids(self, label_arr):
        if self.train:
            # generate positive samples' centroids
            pos_centroids = self._get_pos_centroids(label_arr)

            # generate negative samples' centroids
            neg_centroids = self._get_neg_centroids(pos_centroids,
                label_arr.shape)

            # sample positives and negatives when necessary
            num_pos = len(pos_centroids)
            num_neg = len(neg_centroids)
            if num_pos >= self.num_samples // 2:
                num_neg = self.num_samples - num_pos
            elif num_pos >= self.num_samples:
                num_pos = self.num_samples // 2
                num_neg = self.num_samples // 2

            if num_pos < len(pos_centroids):
                pos_centroids = [pos_centroids[i] for i in np.random.choice(
                    range(0, len(pos_centroids)), size=num_pos, replace=False)]
            if num_neg < len(neg_centroids):
                neg_centroids = [neg_centroids[i] for i in np.random.choice(
                    range(0, len(neg_centroids)), size=num_neg, replace=False)]

            roi_centroids = pos_centroids + neg_centroids
        else:
            roi_centroids = [list(range(0, x, y // 2))[1:-1] + [x - y // 2]
                for x, y in zip(label_arr.shape, self.crop_size)]
            roi_centroids = list(product(*roi_centroids))

        return roi_centroids

    def _crop_roi(self, arr, centroid):
        roi = np.ones(tuple([self.crop_size] * 3)) * (-1024)

        src_beg = [max(0, centroid[i] - self.crop_size // 2)
            for i in range(len(centroid))]
        src_end = [min(arr.shape[i], centroid[i] + self.crop_size // 2)
            for i in range(len(centroid))]
        dst_beg = [max(0, self.crop_size // 2 - centroid[i])
            for i in range(len(centroid))]
        dst_end = [min(arr.shape[i] - (centroid[i] - self.crop_size),
            self.crop_size) for i in range(len(centroid))]
        roi[
            dst_beg[0]:dst_end[0],
            dst_beg[1]:dst_end[1],
            dst_beg[2]:dst_end[2],
        ] = arr[
            src_beg[0]:src_end[0],
            src_beg[1]:src_end[1],
            src_beg[2]:src_end[2],
        ]

        return roi

    def _apply_transforms(self, image):
        for t in self.transforms:
            image = t(image)

        return image

    def __getitem__(self, idx):
        # read image and label
        public_id = self.df.public_id[idx]
        image_path = os.path.join(self.image_dir, f"{public_id}-image.nii.gz")
        label_path = os.path.join(self.label_dir, f"{public_id}-label.nii.gz")
        image = nib.load(image_path)
        label = nib.load(label_path)
        image_arr = image.get_fdata().astype(np.int16)
        label_arr = label.get_fdata().astype(np.uint8)

        # calculate rois' centroids
        roi_centroids = self._get_roi_centroids(label_arr)

        # crop rois
        image_rois = [self._crop_roi(image_arr, centroid)
            for centroid in roi_centroids]
        label_rois = [self._crop_roi(label_arr, centroid)
            for centroid in roi_centroids]

        if self.transforms is not None:
            image_rois = [self._apply_transforms(image_roi)
                for image_roi in image_rois]

        image_rois = torch.tensor(np.stack(image_rois)[:, np.newaxis, ...],
            dtype=torch.float)
        label_rois = torch.tensor(np.stack(label_rois)[:, np.newaxis, ...],
            dtype=torch.float)

        return image_rois, label_rois

    @staticmethod
    def collate_fn(samples):
        image_rois = torch.cat([x[0] for x in samples])
        label_rois = torch.cat([x[1] for x in samples])

        return image_rois, label_rois

    @staticmethod
    def get_dataloader(dataset, batch_size, shuffle=False,
            collate_fn=FracNetNiiDataset.collate_fn):
        return DataLoader(dataset, batch_size, shuffle, collate_fn=collate_fn)
