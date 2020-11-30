import os
from functools import partial
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

import torch.nn as nn
from fastai.basic_train import Learner
from fastai.train import ShowGraph
from fastai.data_block import DataBunch
from matplotlib import pyplot as plt
from torch import optim

from dataset.fracnet_dataset import FracNetTrainDataset
from dataset import transforms as tsfm
from utils.metrics import dice, recall, precision, fbeta_score
from model.unet import UNet
from model.losses import MixLoss, DiceLoss


def main(args):
    train_image_dir = args.train_image_dir
    train_label_dir = args.train_label_dir
    val_image_dir = args.val_image_dir
    val_label_dir = args.val_label_dir

    batch_size = 4
    num_workers = 4
    optimizer = optim.SGD
    criterion = MixLoss(nn.BCEWithLogitsLoss(), 0.5, DiceLoss(), 1)

    thresh = 0.1
    recall_partial = partial(recall, thresh=thresh)
    precision_partial = partial(precision, thresh=thresh)
    fbeta_score_partial = partial(fbeta_score, thresh=thresh)

    model = UNet(1, 1, n=16)
    model = nn.DataParallel(model.cuda())

    transforms = [
        tsfm.Window(-200, 1000),
        tsfm.MinMaxNorm(-200, 1000)
    ]
    ds_train = FracNetNiiDataset(train_image_dir, train_label_dir,
        transforms=transforms)
    dl_train = FracNetNiiDataset.get_dataloader(ds_train, batch_size, False,
        num_workers)
    ds_val = FracNetNiiDataset(val_image_dir, val_label_dir,
        transforms=transforms)
    dl_val = FracNetNiiDataset.get_dataloader(ds_val, batch_size, False,
        num_workers)

    databunch = DataBunch(dl_train, dl_val,
        collate_fn=FracNetNiiDataset.collate_fn)

    learn = Learner(
        databunch,
        model,
        opt_func=optimizer,
        loss_func=criterion,
        metrics=[dice, recall_partial, precision_partial, fbeta_score_partial]
    )

    learn.fit_one_cycle(
        200,
        1e-1,
        pct_start=0,
        div_factor=1000,
        callbacks=[
            ShowGraph(learn),
        ]
    )


if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument("--train_image_dir", required=True,
        help="The training image nii directory.")
    parser.add_argument("--train_label_dir", required=True,
        help="The training label nii directory.")
    parser.add_argument("--val_image_dir", required=True,
        help="The validation image nii directory.")
    parser.add_argument("--val_label_dir", required=True,
        help="The validation label nii directory.")
    args = parser.parse_args()

    main(args)
