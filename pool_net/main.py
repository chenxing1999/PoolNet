from pool_net.pl_modules import BasePoolNetModule
from pool_net.datasets import MyDataset, padding_collate_function
import argparse

from pytorch_lightning.utilities.seed import seed_everything

seed_everything(1)
import os
import numpy as np
import torch
from torch import nn

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import datetime


from torchvision import transforms


def get_train_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", help="Path to train folder")
    parser.add_argument("--train_csv_file")
    parser.add_argument("--val_root", default=None)
    parser.add_argument("--val_csv_file", default=None)

    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--n_gpus", default=1, type=int)
    parser.add_argument("--epochs", default=100, type=int)

    parser.add_argument("--checkpoint_path", default=None)

    args = parser.parse_args()

    if args.val_root is None:
        args.val_root = args.train_root

    return args


def load_flist(csv_path):
    with open(csv_path) as f:
        contents = [pair.split() for pair in f.readlines()]

    return contents


def main():
    args = parse_args()

    train_flist = load_flist(args.train_csv_file)
    val_flist = load_flist(args.val_csv_file)

    train_transform = get_train_transform()
    train_dataset = MyDataset(
        train_flist, args.train_root, transform=train_transform
    )
    val_dataset = MyDataset(val_flist, args.val_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=padding_collate_function,
        shuffle=True,
    )
    val_loader = None
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=padding_collate_function,
    )

    module = BasePoolNetModule()

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=args.n_gpus,
        resume_from_checkpoint=args.checkpoint_path,
    )

    trainer.fit(module, train_loader, val_loader)

    datetime_format = "%d_%m_%Y %H_%M_%S"
    timestamp = datetime.datetime.now().strftime(datetime_format)
    checkpoint_name = f"model_checkpoint_{timestamp}.cptk"

    # trainer.save_checkpoint(f"lightning_logs/{checkpoint_name}")
    torch.save(module.core, f"lightning_logs/{checkpoint_name}")


if __name__ == "__main__":
    main()
