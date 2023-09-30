import os
import torch
import argparse
import torch.nn as nn

from functools import partial
from torch import save

from fastai.basic_train import Learner
from fastai.train import ShowGraph
from fastai.data_block import DataBunch
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
 
    model = UNet(1, 1, first_out_channels=16)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model.cuda())

    transforms = [
        tsfm.Window(-200, 1000),
        tsfm.MinMaxNorm(-200, 1000)
    ]
    ds_train = FracNetTrainDataset(train_image_dir, train_label_dir,
        transforms=transforms)
    dl_train = FracNetTrainDataset.get_dataloader(ds_train, batch_size, False,
        num_workers)
    ds_val = FracNetTrainDataset(val_image_dir, val_label_dir,
        transforms=transforms)
    dl_val = FracNetTrainDataset.get_dataloader(ds_val, batch_size, False,
        num_workers)

    databunch = DataBunch(dl_train, dl_val,
        collate_fn=FracNetTrainDataset.collate_fn)

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

    if args.save_model:
        save(model.module.state_dict(), "./model_weights.pth")


if __name__ == "__main__":
    data_directory = os.path.join(os.getcwd(), "data")
    train_image_dir = os.path.join(data_directory, "train", "ribfrac-train-images")
    train_label_dir = os.path.join(data_directory, "train", "ribfrac-train-labels")
    val_image_dir = os.path.join(data_directory, "val", "ribfrac-val-images")
    val_label_dir = os.path.join(data_directory, "val", "ribfrac-val-labels")

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_image_dir",
        help="The training image nii directory.", default=train_image_dir)
    parser.add_argument("--train_label_dir",
        help="The training label nii directory.", default=train_label_dir)
    parser.add_argument("--val_image_dir",
        help="The validation image nii directory.", default=val_image_dir)
    parser.add_argument("--val_label_dir",
        help="The validation label nii directory.", default=val_label_dir)
    parser.add_argument("--save_model", default=False,
        help="Whether to save the trained model.")
    args = parser.parse_args()

    main(args)
