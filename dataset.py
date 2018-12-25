# -*- coding: utf-8 -*-
import os
from PIL import Image
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor
import numpy as np
import torch


def is_image_file(filename):
	return any (filename.endswith (extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
	y = Image.open (filepath).convert ('RGB')
	return y


class DatasetFromFolder (data.Dataset):
	def __init__(self, image_dir, transform=None):
		super (DatasetFromFolder, self).__init__ ()
		self.image_filenames, self.labels = _get_data(image_dir)
		self.transform = transform

	def __getitem__(self, index):
		inputs = load_img (self.image_filenames[index])
		labels = self.labels[index]
		images = self.transform (inputs)
		labels = binary(labels)
		return images, labels

	def __len__(self):
		return len (self.image_filenames)


def _get_data(image_dir):
	img_paths = []
	labels = []
	for name in os.listdir(image_dir):
		if is_image_file(name):
			img_paths.append(os.path.join(image_dir, name))
			labels.append(int(name.split("_")[0]))
	return img_paths, labels

def binary(label):
	bin_label = torch.zeros((145, ))
	bin_label[label-1] = 1
	return bin_label



def transform():
	return Compose ([ToTensor()])


def get_dataset(image_dir=None):
	return DatasetFromFolder (image_dir, transform=transform ())
