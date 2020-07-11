import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
torch.manual_seed(0)

import cv2
import pandas as pd
import os

import dataset

def load_image(path):
	if os.path.exists(path):
		print(f'File {path} not exists')
		return None
	else:
		im = cv2.imread(path)
		in_ = np.array(im, dtype=np.float32)
		in_ -= np.array((104.00699, 116.66877, 122.67892))
		in_ = in_.transpose((2,0,1))
		return in_

def load_label(path):
	if os.path.exists(path):
		print(f'File {path} not exists')
		return None
	else:
		im = cv2.imread(path)
		label = np.array(im, dtype=np.float32)
		if len(label.shape) == 3:
			label = label[:,:,0]
		label = label / 255.
		label = label[np.newaxis, ...]
		return label

class Dataset(Dataset):
	def __init__(self, csv_file, data_dir, transform=None):
		# Image directory
		self.data_dir = data_dir
		self.csv_file = csv_file
		
		# Load the CSV file contains image info
		with open(self.csv_file, 'r') as f:
			self.images_list = [x.strip() for x in f.readlines()]

		# Number of images in dataset
		self.len = len(self.images_list)

		# The transform is going to be used on image
		self.transform = transform
		
	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		# Image file path
		link = self.images_list[item].split()
		image_name = os.path.join(self.data_dir, link[0])
		gt_name = os.path.join(self.data_dir, link[1])

		# Open image
		image = load_image(image_name)
		label = load_label(gt_name)

		# Transform
		if self.transform:
			image = self.transform(image)
			label = self.transform(label)

		return image, label

def get_loader(config, mode='train', pin=False):
	shuffle = False
	if mode == 'train':
		shuffle = True
		# Transform compose for training
		# transform = transforms.Compose([transforms.CenterCrop(20), transforms.ToTensor()])

		data_loader = data.DataLoader(dataset=Dataset(config.train_root, config.train_list, transform),
						 batch_size=config.batch_size,
						 shuffle=shuffle,
						 num_workers=config.num_thread,
						 pin_memory=pin)
	else:
		# Transform compose for testing
		# transform = transforms.Compose([transforms.CenterCrop(20), transforms.ToTensor()])

		data_loader = data.DataLoader(dataset=Dataset(config.test_root, config.test_list, transform),
						 batch_size=config.batch_size,
						 shuffle=shuffle,
						 num_workers=config.num_thread,
						 pin_memory=pin)
	return data_loader