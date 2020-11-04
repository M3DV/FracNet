import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

from fastai import *
from fastai.basics import *
from fastai.callback import *

from dataset import RibFracDataset
from utils import dice, recall, precision, fbeta_score
from models import UNet, MixLoss, DiceLoss


class PNSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        n = len(self.dataset)
        ix = random.sample(range(n), n)
        pix, nix = [], []
        for i in ix:
            if i < n // 2:
                pix.append(i)
            else:
                nix.append(i)
        ix = zip(pix, nix, random.sample(pix, len(pix)))
        ix = iter(reduce(lambda x, y: x + list(y), ix, []))
        return ix

    def __len__(self):
        return len(self.dataset)


def read_info(path, phase):
    df = pd.read_csv(Path(path, f"seg.csv"))
    df = pd.concat([x[1] for x in df.groupby("neg")])
    if phase == "train":
        df = df[(df["subset"] == 0) | (df["subset"] == 3)]
    else:
        df = df[(df["subset"] == 1) | (df["subset"] == 2)]
    df[["shape", "center"]] = df[["shape", "center"]]\
        .apply(lambda x: x.apply(eval))
    df[["image", "label"]] = df[["image", "label"]]\
        .apply(lambda x: x.apply(lambda y: Path(path, f"{x.name}s", y)))
    return df.reset_index(drop=True)


def checkpoint(root, time):
    def decorator(callback):
        @functools.wraps(callback)
        def wrapper(*args, **kwargs):
            path = Path(root, time)
            path.mkdir(parents=True, exist_ok=True)
            kwargs["root"] = path
            return callback(*args, **kwargs)
        return wrapper
    return decorator


def main(args):
    data_dir = args.data_dir

    batch_size = 96
    workers = 4
    optimizer = optim.SGD
    criterion = MixLoss(nn.BCEWithLogitsLoss(), 0.5, DiceLoss(), 1)

    thresh = 0.1
    recall_partial = partial(recall, thresh=thresh)
    precision_partial = partial(precision, thresh=thresh)
    fbeta_score_partial = partial(fbeta_score, thresh=thresh)

    model = UNet(1, 1, n=16)
    model = nn.DataParallel(model.cuda())

    dataset = {x: RibFracDataset(
        read_info(data_dir, x),
        crop={"size": 64, "scale": 0.5} if x == "train"\
            else {"size": 64, "scale": 0},
        flip=(x == "train")
    ) for x in ["train", "val"]}

    databunch = DataBunch(*[DataLoader(
        dataset[x],
        batch_size=batch_size,
        sampler=PNSampler(dataset[x]),
        num_workers=workers,
        pin_memory=True,
        drop_last=False
    ) for x in ["train", "val"]])

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
    parser.add_argument("--data_dir", type=str, required=True,
        help="The cropped .npz directory.")
    args = parser.parse_args()

    main(args)
