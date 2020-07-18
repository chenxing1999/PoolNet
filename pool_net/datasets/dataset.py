import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image
import os

import logging
logger = logging.getLogger(__name__)


default_transform = transforms.Compose(
    [
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_image(path):
    if os.path.exists(path):
        logger.error(f"File {path} not exists")
        return None
    else:
        im = Image.open(path).convert("RGB")
        return im


def load_label(path):
    if os.path.exists(path):
        logger.error(f"File {path} not exists")
        return None
    else:
        im = Image.open(path)

        return im


class MyDataset(Dataset):
    def __init__(self, flist, data_dir, transform=default_transform):
        """
        Constructer of MyDataset
        
        :param flist: List of image file name and ground truth tuple
        :param data_dir: directory of folder containing all images
        :param transform: transform function will be used on images
        :return: returns nothing
        """
        # Image directory
        self.data_dir = data_dir

        # images_list: List of tuple(path to img, path to gt) 
        self.images_list = flist

        # Number of images in dataset
        self.len = len(self.images_list)

        # The transform is going to be used on image
        self.transform = transform

    def __len__(self):
        """
        return number of images
        """
        return self.len

    def __getitem__(self, idx):
        """
        Getter
        
        :param idx: the index of image
        :return: return the i'th image and its label
        """
        # Image file path
        link = self.images_list[idx]
        image_name = os.path.join(self.data_dir, link[0])
        gt_name = os.path.join(self.data_dir, link[1])

        # Open image
        image = Image.open(image_name).convert("RGB")
        label = Image.open(gt_name).convert("L")

        # Transform
        if self.transform:
            image = self.transform(image)
            label = transforms.ToTensor()(label)

        return image, label


def get_loader(config, mode="train", pin=False):
    shuffle = False
    if mode == "train":
        shuffle = True
        # Transform compose for training
        transform = default_transform

        data_loader = data.DataLoader(
            dataset=myDataset(config.train_root, config.train_list, transform),
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_thread,
            pin_memory=pin,
        )
    else:
        # Transform compose for testing
        transform = default_transform

        data_loader = data.DataLoader(
            dataset=myDataset(config.test_root, config.test_list, transform),
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_thread,
            pin_memory=pin,
        )
    return data_loader
