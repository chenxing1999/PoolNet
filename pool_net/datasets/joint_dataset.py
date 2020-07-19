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



class EdgeDataset(Dataset):
    def __init__(self, sal_flist, sal_data_dir, edge_flist, edge_data_dir, transform=default_transform):
        """
        Constructer of MyDataset
        
        :param sal_flist: List of salient image file name and ground truth tuple
        :param sal_data_dir: directory of folder containing all salient images
        :param edge_flist: List of edge image file name and ground truth tuple
        :param edge_data_dir: directory of folder containing all edge images
        :param transform: transform function will be used on images
        :return: returns nothing
        """
        # Salient Image directory
        self.sal_data_dir = sal_data_dir

        # sal_images_list: List of salient image tuple(path to img, path to gt) 
        self.sal_images_list = sal_flist

        # Edge Image directory
        self.sal_data_dir = edge_data_dir

        # edge_images_list: List of edge image tuple(path to img, path to gt) 
        self.edge_images_list = edge_flist
        
        # Number of images in dataset
        self.len = len(self.sal_images_list)

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
        :return: return the i'th salient image, its label, i'th edge image and edge image label 
        """
        # Salient Image file path
        salient_link = self.salient_images_list[idx]
        salient_image_name = os.path.join(self.salient_data_dir, salient_link[0])
        salient_gt_name = os.path.join(self.salient_data_dir, salient_link[1])

        # Edge Image file path
        edge_link = self.edge_images_list[idx]
        edge_image_name = os.path.join(self.edge_data_dir, edge_link[0])
        edge_gt_name = os.path.join(self.edge_data_dir, edge_link[1])
        
        # Open salient image
        salient_image = Image.open(salient_image_name).convert("RGB")
        salient_label = Image.open(salient_gt_name).convert("L")
        # Open edge image
        edge_image = Image.open(edge_image_name).convert("RGB")
        edge_label = Image.open(edge_gt_name).convert("L")

        # Transform
        if self.transform:
            salient_image = self.transform(salient_image)
            salient_label = transforms.ToTensor()(salient_label)
            
            edge_image = self.transform(edge_image)
            edge_label = transforms.ToTensor()(edge_label)

        return salient_image, salient_label, edge_image, edge_label