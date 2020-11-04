import os
import re

import nibabel as nib
from fastai.basics import *


class RibFracDataset(Dataset):
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


class NiiDataset:
    """
    A dataloder reading all .nii files under one directory.

    Parameters
    ----------
    root_dir : str
        The directory where all .nii files reside.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = sorted([os.path.join(root_dir, x)
            for x in os.listdir(root_dir)
            if x.endswith(".nii") or x.endswith(".nii.gz")])
        # use regular expression to accomodate both .nii and .nii.gz
        self.pid_list = [re.sub(r"(\.nii)|(\.gz)|(-label)", "",
            os.path.basename(x)) for x in self.file_list]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        array = nib.load(self.file_list[idx]).get_fdata()

        return array, self.pid_list[idx]
