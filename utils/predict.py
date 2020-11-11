from itertools import product

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from fastprogress.fastprogress import progress_bar


class Slicer(Dataset):
    def __init__(self, image, size, centers=None):
        '''image: (C, D, H, W) '''
        self.size = size
        self.image = image
        self.centers = self.center(image, size) if centers is None else centers
        self.indices = [None] * len(self.centers)

    def __getitem__(self, idx):
        index = self.crop(self.image, self.centers[idx], self.size)
        image = self.image[index].astype(np.float32)
        self.indices[idx] = index

        return image, idx

    def __len__(self):
        return len(self.centers)

    def crop(self, image, center, size):
        '''image: (C, D, H, W) '''
        shape = image.shape[1:]
        minp1 = [x - y // 2 for x, y in zip(center, size)]
        minp2 = [x if x > 0 else 0 for x in minp1]

        maxp1 = [x + y for x, y in zip(minp2, size)]
        maxp2 = [x if x < y else y for x, y in zip(maxp1, shape)]

        minp3 = [x - y for x, y in zip(maxp2, size)]
        index = [slice(x, y) for x, y in zip(minp3, maxp2)]
        index = tuple([slice(None)] + index)

        return index

    def center(self, image, size):
        '''image: (C, D, H, W) '''
        shape = image.shape[1:]
        coord = [list(range(0, x, y // 2))[1:-1] + [x - y // 2] for x, y in zip(shape, size)]
        coord = list(product(*coord))

        return coord


class Predictor(object):
    def __init__(self, model, size):
        self.size = size
        self.model = model

    def __call__(self, image, centers=None, mbar=None, **kwargs):
        '''image: (C, D, H, W) '''
        slicer = Slicer(image, self.size, centers)
        loader = DataLoader(slicer, **kwargs)
        result = torch.zeros(image.shape).cuda() if centers is None else [None] * len(centers)

        for (inputs, indices) in progress_bar(loader, parent=mbar):
            inputs = inputs.cuda()

            with torch.set_grad_enabled(False):
                preds = self.model(inputs).sigmoid()

            for (pred, idx) in zip(preds, indices):
                if centers is None:
                    index = slicer.indices[idx]; block = result[index]
                    result[index] = torch.where(block > 0, (block + pred) / 2, pred)
                else:
                    result[idx] = pred.item()

        return result
