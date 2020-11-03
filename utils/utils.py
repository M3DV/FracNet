import SimpleITK as sitk
from fastai.basics import *


def imread(path):
    path = Path(path)
    if path.is_dir():
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(path)))
        return reader.Execute()
    elif path.is_file():
        return sitk.ReadImage(str(path))
    else:
        raise NotImplementedError('The path must be a file or dicom dir')


def imcrop(image, center, size):
    minp1 = [x - y // 2 for x, y in zip(center, size)]
    minp2 = [x if x > 0 else 0 for x in minp1]

    maxp1 = [x + y for x, y in zip(minp2, size)]
    maxp2 = [x if x < y else y for x, y in zip(maxp1, image.shape)]

    minp3 = [x - y for x, y in zip(maxp2, size)]
    index = [slice(x, y) for x, y in zip(minp3, maxp2)]
    block = image[tuple(index)]

    return block


def resample(image, spacing=(1, 1, 1), label=False):
    zoom = [x / y for x, y in zip(image.GetSpacing(), spacing)]
    size = [int(round(x * y)) for x, y in zip(image.GetSize(), zoom)]

    sampler = sitk.ResampleImageFilter()
    sampler.SetSize(size)
    sampler.SetOutputSpacing(spacing)
    sampler.SetOutputOrigin(image.GetOrigin())
    sampler.SetOutputDirection(image.GetDirection())

    if label:
        sampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        sampler.SetInterpolator(sitk.sitkBSpline)

    return sampler.Execute(image)
